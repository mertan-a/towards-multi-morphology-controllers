import os
import random
import numpy as np
import multiprocessing
import torch

from utils import prepare_rundir
from population import POPULATION, QD_ARCHIVE
from algorithms import AFPO, MAP_ELITES
from make_gif import MAKEGIF

import argparse
parser = argparse.ArgumentParser(description='run jobs')

# task 
parser.add_argument('--task', '-t', help='specify the task',
                    choices=['Walker-v0', 'BridgeWalker-v0'], default='Walker-v0')
# evolutionary algorithm related arguments
parser.add_argument('--evolutionary_algorithm', '-ea', type=str,
                    choices=['afpo', 'qd'], help='choose the evolutionary algorithm')
parser.add_argument('-nrp', '--nr_parents', type=int,
                     help='number of parents')
parser.add_argument('-nrg', '--nr_generations', type=int,
                     help='number of generations')
parser.add_argument('--nr_random_individual', '-nri', type=int, 
                    help='Number of random individuals to insert each generation')
parser.add_argument('-jt', '--job_type', type=str, 
                    help='job type', choices=['optimizeBrain', 'cooptimize'])
# softrobot related arguments
parser.add_argument('--use_fixed_body', '-ufbo',
                    help='use fixed body/ies', action='store_true')
parser.add_argument('--fixed_bodies', '-fbo', nargs='+',
                    help='specify the fixed body/ies', type=str,  choices=['biped', 'worm', 'triped', 'block', 'deneme'])
parser.add_argument('--fixed_body_path', '-fbp',
                    help='specify the path to the individual that contains the body you want', default=None)
parser.add_argument('--bounding_box', '-bb', nargs='+', type=int,
                    help='Bounding box dimensions (x,y). e.g.IND_SIZE=(6, 6)->workspace is a rectangle of 6x6 voxels') # trying to get rid of this
parser.add_argument('--use_pretrained_brain', '-uptbr',
                    help='use pretrained brain', action='store_true')
parser.add_argument('--pretrained_brain', '-ptbr',
                    help='specify the path to the pretrained brain\'s pkl')
parser.add_argument('--controller', '-ctrl', help='specify the controller',
                    choices=['DECENTRALIZED', 'CENTRALIZED', 'TRANSFORMER'], default='DECENTRALIZED')

parser.add_argument('-id', '--id', type=int, default=1,
                    help='id of the job')
parser.add_argument('--gif_every', '-ge', type=int, default=50)

args = parser.parse_args()

def run(args):

    multiprocessing.set_start_method('spawn')

    # run the job directly
    args.rundir = prepare_rundir(args)
    print('rundir', args.rundir)

    # if this experiment is currently running or has finished, we don't want to run it again
    if os.path.exists(args.rundir + '/RUNNING'):
        print('Experiment is already running')
        exit()
    if os.path.exists(args.rundir + '/FINISHED'):
        print('Experiment has already finished')
        exit()

    # Initializing the random number generator for reproducibility
    SEED = args.id
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Setting up the optimization algorithm and runnning
    if args.evolutionary_algorithm == 'afpo':
        pareto_optimization = AFPO(args=args, population=POPULATION(args=args))
        pareto_optimization.optimize()
    elif args.evolutionary_algorithm == 'qd':
        map_elites = MAP_ELITES(args=args, map=QD_ARCHIVE(args=args))
        map_elites.optimize()
    else:
        raise ValueError('unknown evolutionary algorithm')

    # delete running file
    if os.path.isfile(args.rundir + '/RUNNING'):
        os.remove(args.rundir + '/RUNNING')

    # write a file to indicate that the job finished successfully
    with open(args.rundir + '/FINISHED', 'w') as f:
        pass

if __name__ == '__main__':

    # sanity checks
    if args.job_type == 'cooptimize' and args.use_fixed_body == True:
        raise ValueError('cooptimization is not supported for fixed bodies')
    if args.evolutionary_algorithm == 'qd':
        assert args.job_type == 'cooptimize', 'qd is only supported for cooptimization'

    # run the job
    run(args)





