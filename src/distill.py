import numpy as np
import os
import _pickle as pickle
import torch
from argparse import Namespace
import imageio
from pygifsicle import optimize

from networks import NeuralNetwork, TransformerNetwork, NeuralNetworkModular
from body import FIXED_BODY
from brain import CENTRALIZED
from individual import INDIVIDUAL
from simulator import make_env
from evogym_wrappers import RenderWrapper

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='biped.pkl', nargs='+',
                    help='name of the dataset to be distilled, assuming they are in ../datasets/')
parser.add_argument('--network', type=str, default='simple', choices=['simple', 'transformer', 'modular'],
                    help='network architecture')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--nr_steps', type=int, default=10000,
                    help='number of training steps')
parser.add_argument('--val_every', type=int, default=100,
                    help='validate every this many steps')
parser.add_argument('--save_to', type=str, default='distilled_brain.pt',
                    help='name to save the distilled brain, assuming they are saved in ../distilled_controllers/')
args = parser.parse_args()


def simulate(ind, args, body_name=None, measure_fitness=False):
    from time import sleep
    # get the env
    env = make_env(ind.body.bodies[0]['structure'], **vars(args))
    if measure_fitness == False:
        env = RenderWrapper(env, render_mode='img')
    cum_rewards = []
    for i in range(20):
        # record keeping
        cum_reward = 0
        # run simulation
        obs = env.reset()
        for t in range(500):
            # apply noise
            obs += np.random.normal(0, 0.01, size=obs.shape)
            # collect actions
            actions = ind.brain.get_action(obs)
            # apply noise
            actions += np.random.normal(0, 0.01, size=actions.shape)
            # step
            obs, r, d, i = env.step(actions)
            # record keeping
            cum_reward += r
            # break if done
            if d:
                break
            #if measure_fitness == False:
            #    sleep(0.05)
        cum_rewards.append(cum_reward)
        if measure_fitness == False:
            imageio.mimsave(f"results/{body_name}_distilled_controller.gif", env.imgs, duration=1/200000.0)
            break
    return (np.mean(cum_rewards), np.std(cum_rewards))

def train(datasets, network, hyperparams):
    """ trains the network on the datasets 

    Parameters
    ----------
    datasets : list of datasets
        each dataset is a dictionary with keys 'obs_train', 'act_train', 'obs_val', 'act_val'
        sample items from each dataset in each batch

    network : NeuralNetwork
        the network to be trained
        assuming the input and output sizes are compatible with the datasets

    hyperparams : dict
        hyperparameters for training
        'batch_size' : int
            batch size
        'nr_steps' : int
            number of training steps
        'val_every' : int
            validate every this many steps

    Returns
    -------
    losses : list of floats
        losses for each training step

    """
    # optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    # train the network
    losses = []
    val_losses = []
    for i in range(hyperparams['nr_steps']):
        # get a random batch
        batch_size = hyperparams['batch_size']
        X = []
        Y = []
        for dataset in datasets:
            idx = np.random.randint(0, len(dataset['obs_train']), batch_size//len(datasets))
            X.append(dataset['obs_train'][idx])
            Y.append(dataset['act_train'][idx])
        # concatenate
        obs = np.concatenate(X, axis=0)
        act = np.concatenate(Y, axis=0)
        # convert to tensors
        obs = torch.from_numpy(obs).double()
        act = torch.from_numpy(act).double()
        # train
        est_act = network(obs)
        loss = torch.mean((est_act - act)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # record keeping
        losses.append(loss.item())
        # validate
        if (i+1) % hyperparams['val_every'] == 0:
            val_loss = validate(network, datasets, hyperparams)
            print('Loss:', np.mean(losses[-hyperparams['val_every']:]), 'Val loss:', val_loss)
            val_losses.append(val_loss)

    return losses, val_losses

def validate(network, datasets, hyperparams):
    """ validates the network on the datasets

    Parameters
    ----------
    network : NeuralNetwork
        the network to be validated
        assuming the input and output sizes are compatible with the datasets

    datasets : list of datasets
        each dataset is a dictionary with keys 'obs_train', 'act_train', 'obs_val', 'act_val'
        sample items from each dataset in each batch

    hyperparams : dict
        hyperparameters for training
        'batch_size' : int
            batch size
        'nr_steps' : int
            number of training steps
        'val_every' : int
            validate every this many steps

    Returns
    -------
    val_loss : float
        validation loss

    """
    # get a random batch
    batch_size = hyperparams['batch_size']
    X = []
    Y = []
    for dataset in datasets:
        idx = np.random.randint(0, len(dataset['obs_val']), batch_size//len(datasets))
        X.append(dataset['obs_val'][idx])
        Y.append(dataset['act_val'][idx])
    # concatenate
    obs = np.concatenate(X, axis=0)
    act = np.concatenate(Y, axis=0)
    # convert to tensors
    obs = torch.from_numpy(obs).double()
    act = torch.from_numpy(act).double()
    # turn off gradients
    with torch.no_grad():
        network.eval()
        # validate
        est_act = network(obs)
        val_loss = torch.mean((est_act - act)**2)
        network.train()
    return val_loss.item()

def main(args):
    # read the datasets
    datasets = []
    for name in args.datasets:
        print('Reading', name)
        with open(f'../datasets/{name}', 'rb') as f:
            dataset = pickle.load(f)
        datasets.append(dataset)
    # get the network
    example_obs = datasets[0]['obs_train'][0]
    example_act = datasets[0]['act_train'][0]
    if args.network == 'simple':
        nn = NeuralNetwork(input_size=example_obs.shape[0], output_size=example_act.shape[0])
    elif args.network == 'modular':
        nn = NeuralNetworkModular(input_size=example_obs.shape[0], output_size=example_act.shape[0])
    else:
        nn = TransformerNetwork(robot_bounding_box=[5,5], observation_per_voxel_size=9,
                                embedding_size=64, num_attention_heads=8, num_encoder_layers=4)
    nn.double()
    # optimizer
    optimizer = torch.optim.Adam(nn.parameters(), lr=1e-3)
    # hyperparams
    hyperparams = {'batch_size': args.batch_size, 'nr_steps': args.nr_steps, 'val_every': args.val_every}

    # train the network
    losses, val_losses = train(datasets, nn, hyperparams=hyperparams)
    # plot the losses and val losses
    import matplotlib.pyplot as plt
    plt.plot(losses, label='loss')
    plt.plot(np.arange(100, len(losses)+1, 100), val_losses, label='val loss')
    plt.savefig(f'../distilled_controllers/{args.save_to}.png')
    plt.close()
    # save the network
    if args.network == 'simple' or args.network == 'modular':
        torch.jit.save(torch.jit.script(nn), f'../distilled_controllers/{args.save_to}')
    else:
        torch.save(nn, f'../distilled_controllers/{args.save_to}')

if __name__ == "__main__":
    main(args)



        
