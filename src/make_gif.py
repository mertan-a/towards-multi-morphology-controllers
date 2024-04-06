import gym
import numpy as np
np.float = float
import matplotlib.pyplot as plt
from pygifsicle import optimize
import imageio
import _pickle as pickle

from population import POPULATION
from evogym.envs import *
from evogym_wrappers import RenderWrapper, ActionSkipWrapper, ActionSpaceCorrectionWrapper, LocalObservationWrapper, GlobalObservationWrapper, LocalActionWrapper, GlobalActionWrapper, RewardShapingWrapper, TransformerObservationWrapper, TransformerActionWrapper

class MAKEGIF():

    def __init__(self, args):
        self.kwargs = vars(args)
        # read the ind from pkl
        with open(self.kwargs['path_to_ind'], 'rb') as f:
            # unpickle and check if it is a list
            unpickled = pickle.load(f)
            if isinstance(unpickled, POPULATION) or isinstance(unpickled, list):
                self.ind = unpickled[0]
            else:
                self.ind = unpickled

    def run(self):
        for b in self.ind.body.bodies:
            env = gym.make(self.kwargs['task'], body=b['structure'], connections=get_full_connectivity(b['structure']))
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = RenderWrapper(env, render_mode='img')
            env = ActionSkipWrapper(env, skip=5)
            env = ActionSpaceCorrectionWrapper(env)
            if self.kwargs['controller'] in ['DECENTRALIZED']:
                env = LocalObservationWrapper(env, **self.kwargs)
                env = LocalActionWrapper(env, **self.kwargs)
            elif self.kwargs['controller'] in ['CENTRALIZED']:
                env = GlobalObservationWrapper(env, **self.kwargs)
                env = GlobalActionWrapper(env, **self.kwargs)
            elif self.kwargs['controller'] in ['TRANSFORMER']:
                env = TransformerObservationWrapper(env, **self.kwargs)
                env = TransformerActionWrapper(env, **self.kwargs)
            else:
                raise ValueError('Unknown controller', self.kwargs['controller'])
            env = RewardShapingWrapper(env)
            env.seed(17)
            env.action_space.seed(17)
            env.observation_space.seed(17)
            env.env.env.env.env.env.env.env._max_episode_steps = 500

            # run the environment
            cum_reward = 0
            observation = env.reset()
            for ts in range(500):
                action = self.ind.brain.get_action(observation)
                observation, reward, done, _ = env.step(action)
                cum_reward += reward
                if type(done) == bool:
                    if done:
                        break
                elif type(done) == np.ndarray:
                    if done.all():
                        break
                else:
                    raise ValueError('Unknown type of done', type(d))
            # print the env.imgs[0] to see what it looks like
            # don't print axis and remove white space
            #img = env.imgs[0]
            #plt.axis('off')
            #plt.imshow(img)
            #plt.savefig(self.kwargs['output_path'] + '.png', bbox_inches='tight', pad_inches=0)
            #plt.close()
            imageio.mimsave(f"{self.kwargs['output_path']}_{cum_reward}_{b['name']}.gif", env.imgs, duration=(1/50.0))
            try:
                optimize(f"{self.kwargs['output_path']}_{cum_reward}_{b['name']}")
            except:
                pass

