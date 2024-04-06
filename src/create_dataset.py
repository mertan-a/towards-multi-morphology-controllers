import numpy as np
import os
import _pickle as pickle
from argparse import Namespace
import time

from simulator import make_env
from evogym_wrappers import RenderWrapper

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='Walker-v0')
parser.add_argument('--controller', type=str, default='CENTRALIZED', # right now only centralized is supported
                    help='controller type in the archive')
parser.add_argument('--path_to_archive', type=str, default=None, 
                    help='path to the archive') 
parser.add_argument('--name_of_dataset', type=str, default=None,
                    help='name of the dataset to be created')
parser.add_argument('--nr_runs', type=int, default=100,
                    help='number of runs to be simulated for each individual')
parser.add_argument('--train_val_split', type=float, default=0.8,
                    help='percentage of the dataset to be used for training')
args = parser.parse_args()

def create_dataset(args):
    """ create dataset from the individuals """
    with open(args.path_to_archive, 'rb') as f:
        archive = pickle.load(f)
        archive = archive.map

    datasets = []
    for ind in archive.values():
        if ind is None:
            continue
        # create dataset for the individual
        for i in range(args.nr_runs):
            print(i)
            dataset = simulate(ind, args)
            datasets.append(dataset)
    # merge different individuals
    obs = []
    act = []
    for dataset in datasets:
        assert len(dataset['obs']) == len(dataset['act'])
        obs.extend(dataset['obs'])
        act.extend(dataset['act'])
    # train-val split
    obs = np.array(obs)
    act = np.array(act)
    idx = np.arange(obs.shape[0])
    np.random.shuffle(idx)
    split = int(args.train_val_split * obs.shape[0])
    obs_train = obs[idx[:split]]
    act_train = act[idx[:split]]
    obs_val = obs[idx[split:]]
    act_val = act[idx[split:]]
    # save the datasets
    to_save = {'obs_train': obs_train, 'act_train': act_train, 'obs_val': obs_val, 'act_val': act_val}
    path = os.path.join("../datasets", args.name_of_dataset + "_" + args.controller + ".pkl")
    with open(path, 'wb') as f:
        pickle.dump(to_save, f, protocol=-1)
    # process the dataset for transformer and modular
    process_dataset_for_transformer(path)
    process_dataset_for_modular(path)

def simulate(ind, args):
    """ assuming that there is only one body of interest in the individual """
    # get the env
    env = make_env(ind.body.bodies[0]['structure'], **vars(args))
    # run simulation
    dataset = { 'obs': [], 'act': [] }
    obs = env.reset()
    dataset['obs'].append(obs)
    for t in range(500):
        # collect actions
        actions = ind.brain.get_action(obs)
        dataset['act'].append(actions)
        # step
        obs, r, d, i = env.step(actions)
        # break if done
        if d:
            break
        dataset['obs'].append(obs)
    return dataset

def process_dataset_for_transformer(dataset_path):
    """takes a path to a dataset
    process the observations and actions to be used by the transformer network model"""
    print(f"Processing dataset for transformer: {dataset_path}")
    # read the dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    # process observation and action pairs
    transformer_training_observations = []
    for obs in dataset['obs_train']:
        # process them
        transformer_obs = process_obs_for_transformer(obs)
        # append to the list
        transformer_training_observations.append(transformer_obs)
    transformer_training_observations = np.array(transformer_training_observations)
    # do it for the validation set
    transformer_validation_observations = []
    for obs in dataset['obs_val']:
        # process them
        transformer_obs = process_obs_for_transformer(obs)
        # append to the list
        transformer_validation_observations.append(transformer_obs)
    transformer_validation_observations = np.array(transformer_validation_observations)
    # save the processed dataset with the same name, but with _transformer suffix and to qd_datasets_transformer folder
    to_save = {'obs_train': transformer_training_observations, 'act_train': dataset['act_train'], 'obs_val': transformer_validation_observations, 'act_val': dataset['act_val']}
    base, name = os.path.split(dataset_path)
    path = os.path.join( base, name.split('_')[0]+'_transformer.pkl' )
    print(f"Saving processed dataset for transformer: {path}")
    with open(path, 'wb') as f:
        pickle.dump(to_save, f, protocol=-1)

def process_obs_for_transformer(obs):
    """takes an observation and
    processes them to be used by the transformer network model"""
    # process the observation
    time = obs[-1]
    inp = []  # inp should be: (num_existing_voxels, observation_per_voxel_size) ### NOPE, it will be (num_voxels, observation_per_voxel_size)
    for i in range(25): #### assuming 5x5 individuals
        voxel_obs = obs[i*8:i*8+8]
        voxel_obs = np.concatenate((voxel_obs, [time]))
        inp.append(voxel_obs)
    inp = np.array(inp)
    return inp

def process_dataset_for_modular(dataset_path):
    """takes a path to a dataset
    process the observations and actions to be used by modular control paradigm """
    print(f"Processing dataset for modular: {dataset_path}")
    # read the dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    ## to do some sanity checks, read the individual and its body
    #with open("inds".join(dataset_path.rsplit('datasets', 1)), 'rb') as f:
    #    ind = pickle.load(f)
    #print(ind.body.bodies[0]['structure'])
    # process observation and action pairs
    modular_training_observations = []
    modular_training_actions = []
    for obs, act in zip(dataset['obs_train'], dataset['act_train']):
        # process them
        modular_obs, modular_act = process_obs_act_for_modular(obs, act)
        # append to the list
        modular_training_observations.extend(modular_obs)
        modular_training_actions.extend(modular_act)
    modular_training_observations = np.array(modular_training_observations)
    modular_training_actions = np.array(modular_training_actions)
    modular_training_actions = np.expand_dims(modular_training_actions, axis=1)
    # do it for the validation set
    modular_validation_observations = []
    modular_validation_actions = []
    for obs, act in zip(dataset['obs_val'], dataset['act_val']):
        # process them
        modular_obs, modular_act = process_obs_act_for_modular(obs, act)
        # append to the list
        modular_validation_observations.extend(modular_obs)
        modular_validation_actions.extend(modular_act)
    modular_validation_observations = np.array(modular_validation_observations)
    modular_validation_actions = np.array(modular_validation_actions)
    modular_validation_actions = np.expand_dims(modular_validation_actions, axis=1)
    # save the processed dataset with the same name, but with _modular suffix and to qd_datasets_modular folder
    to_save = {'obs_train': modular_training_observations, 'act_train': modular_training_actions, 'obs_val': modular_validation_observations, 'act_val': modular_validation_actions}
    base, name = os.path.split(dataset_path)
    path = os.path.join( base, name.split('_')[0]+'_modular.pkl' )
    print(f"Saving processed dataset for modular: {path}")
    with open(path, 'wb') as f:
        pickle.dump(to_save, f, protocol=-1)

def process_obs_act_for_modular(obs, act):
    """takes an observation and
    processes them to be used by the modular control paradigm"""
    # process the observation
    time = obs[-1]
    raw_observations = []
    exists = []
    is_active = []
    for i in range(25): #### assuming 5x5 individuals
        per_voxel_obs = obs[i*8:i*8+8]
        #per_voxel_obs = np.concatenate((per_voxel_obs, [time]))
        raw_observations.append(per_voxel_obs)
        if np.abs(per_voxel_obs[0]) < 0.2: # because observations are noisy, they might not be exactly 0
            exists.append(True)
        else:
            exists.append(False)
        if np.abs(per_voxel_obs[3]) > 0.8 or np.abs(per_voxel_obs[4]) > 0.8:
            is_active.append(True)
        else:
            is_active.append(False)
    raw_observations = np.array(raw_observations).reshape(5,5,8)
    exists = np.array(exists).reshape(5,5)
    is_active = np.array(is_active).reshape(5,5)
    # now we have basic observations for each voxel, now we have to process them for modular observation
    processed_observations = []
    for i in range(25):
        x,y = two_d_idx_of(i)
        if is_active[x,y] == False:
            continue
        obs = []
        # get the neighbors
        neighbors = get_moore_neighbors(x, y)
        # get the volumes of the neighbors
        for neighbor in neighbors:
            if neighbor[0] == -1: # neighbor is only -1 when it is out of bounds
                # observe structure
                neighbor_obs = np.zeros(8)
                neighbor_obs[0] = 1
            else:
                # neighbor exists, lets get its observation from raw_observations
                neighbor_obs = raw_observations[neighbor[1][0], neighbor[1][1]]
            obs.append(neighbor_obs)
        obs = np.array(obs).flatten()
        # observe time
        obs = np.concatenate((obs, [time]))
        processed_observations.append(obs)
    # process the action
    processed_actions = []
    for i in range(25):
        x,y = two_d_idx_of(i)
        if is_active[x,y] == False:
            continue
        processed_actions.append(act[i])
    return processed_observations, processed_actions

def get_moore_neighbors(x, y):
    '''
    returns the 8 neighbors of a voxel in the structure
    '''
    observation_range = 1
    neighbors = []
    min = observation_range * -1
    max = observation_range + 1
    for i in range(min, max):
        for j in range(min, max):
            if x+i >= 0 and x+i < 5 and y+j >= 0 and y+j < 5:
                neighbors.append( (1.0, [x+i, y+j]) )
            else:
                neighbors.append( (-1.0, None) )
    return neighbors

def two_d_idx_of(idx):
    '''
    returns 2d index of a 1d index
    '''
    return idx // 5, idx % 5

def main(args):
    create_dataset(args)

if __name__ == "__main__":
    main(args)






