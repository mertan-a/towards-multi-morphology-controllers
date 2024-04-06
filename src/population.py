import operator
import numpy as np

from individual import INDIVIDUAL
from body import FIXED_BODY, BASIC_BODY
from brain import DECENTRALIZED, CENTRALIZED


class POPULATION(object):
    """A population of individuals"""

    def __init__(self, args):
        """Initialize a population of individuals.

        Parameters
        ----------
        args : object
            arguments object

        """
        self.args = args
        self.individuals = []
        self.non_dominated_size = 0

        while len(self) < self.args.nr_parents:
            self.add_individual()

    def add_individual(self):
        valid = False
        while not valid:
            # body
            if self.args.use_fixed_body:
                if (self.args.fixed_bodies is None or len(self.args.fixed_bodies) == 0) and self.args.fixed_body_path is None:
                    raise ValueError("No fixed bodies specified")
                elif self.args.fixed_bodies is not None and len(self.args.fixed_bodies) > 0:
                    body = FIXED_BODY(fixed_body=self.args.fixed_bodies)
                elif self.args.fixed_body_path is not None:
                    body = FIXED_BODY(fixed_body_path=self.args.fixed_body_path)
                else:
                    raise ValueError("something surely wrong")
            else:
                body = BASIC_BODY(self.args)
            # brain
            if self.args.controller == 'DECENTRALIZED':
                brain = DECENTRALIZED(args=self.args)
            elif self.args.controller == 'CENTRALIZED':
                brain = CENTRALIZED(args=self.args)
            elif self.args.controller == 'TRANSFORMER':
                brain = TRANSFORMER(args=self.args)
            else:
                raise ValueError("Unknown brain type", self.args.controller)
            ind = INDIVIDUAL(body=body, brain=brain)
            if ind.is_valid():
                self.individuals.append(ind)
                valid = True

    def produce_offsprings(self):
        """Produce offspring from the current population."""
        offspring = []
        for counter, ind in enumerate(self.individuals):
            offspring.append(ind.produce_offspring())
        self.individuals.extend(offspring)

    def calc_dominance(self):
        """Determine which other individuals in the population dominate each individual."""

        # if tied on all objectives, give preference to newer individual
        self.sort(key="age", reverse=False)

        # clear old calculations of dominance
        self.non_dominated_size = 0
        for ind in self:
            ind.dominated_by = []
            ind.pareto_level = 0

        for ind in self:
            for other_ind in self:
                if other_ind.self_id != ind.self_id:
                    if self.dominated_in_multiple_objectives(ind, other_ind) and (ind.self_id not in other_ind.dominated_by):
                        ind.dominated_by += [other_ind.self_id]

            ind.pareto_level = len(ind.dominated_by)  # update the pareto level

            # update the count of non_dominated individuals
            if ind.pareto_level == 0:
                self.non_dominated_size += 1

    def dominated_in_multiple_objectives(self, ind1, ind2):
        """Calculate if ind1 is dominated by ind2 according to all objectives in objective_dict.

        If ind2 is better or equal to ind1 in all objectives, and strictly better than ind1 in at least one objective.

        """
        wins = []  # 1 dominates 2
        wins += [ind1.fitness > ind2.fitness]
        wins += [ind1.age < ind2.age]
        return not np.any(wins)

    def sort_by_objectives(self):
        """Sorts the population multiple times by each objective, from least important to most important."""
        self.sort(key="age", reverse=False)
        self.sort(key="fitness", reverse=True)

        self.sort(key="pareto_level", reverse=False)  # min

    def update_ages(self):
        """Increment the age of each individual."""
        for ind in self:
            ind.age += 1

    def sort(self, key, reverse=False):
        """Sort individuals by their attributes.

        Parameters
        ----------
        key : str
            An individual-level attribute.

        reverse : bool
            True sorts from largest to smallest (useful for maximizing an objective).
            False sorts from smallest to largest (useful for minimizing an objective).

        """
        return self.individuals.sort(reverse=reverse, key=operator.attrgetter(key))

    def __iter__(self):
        """Iterate over the individuals. Use the expression 'for n in population'."""
        return iter(self.individuals)

    def __contains__(self, n):
        """Return True if n is a SoftBot in the population, False otherwise. Use the expression 'n in population'."""
        try:
            return n in self.individuals
        except TypeError:
            return False

    def __len__(self):
        """Return the number of individuals in the population. Use the expression 'len(population)'."""
        return len(self.individuals)

    def __getitem__(self, n):
        """Return individual n.  Use the expression 'population[n]'."""
        return self.individuals[n]

    def pop(self, index=None):
        """Remove and return item at index (default last)."""
        return self.individuals.pop(index)

    def append(self, individuals):
        """Append a list of new individuals to the end of the population.

        Parameters
        ----------
        individuals : list of/or INDIVIDUAL
            A list of individuals to append or a single INDIVIDUAL to append

        """
        if type(individuals) == list:
            for n in range(len(individuals)):
                if type(individuals[n]) != INDIVIDUAL:
                    raise TypeError("Non-INDIVIDUAL added to the population")
            self.individuals += individuals
        elif type(individuals) == INDIVIDUAL:
            self.individuals += [individuals]
        else:
            raise TypeError("Non-INDIVIDUAL added to the population")


class QD_ARCHIVE():
    """A population of individuals to be used with MAP-Elites"""

    def __init__(self, args):
        """Initialize a population of individuals.

        Parameters
        ----------
        args : object
            arguments object

        """
        self.args = args
        self.map = {}
        # define the bins. for now we assume that the feature space is 2D based on the number of existing voxels and percentage of active voxels
        self.n_bins_existing_voxels = self.args.bounding_box[0] * self.args.bounding_box[1]
        self.n_bins_active_voxels = self.args.bounding_box[0] * self.args.bounding_box[1]
        for i in range(1, self.n_bins_existing_voxels+1):
            for j in range(1, self.n_bins_active_voxels+1):
                self.map[(i,j)] = None

    def get_random_individual(self):
        valid = False
        while not valid:
            # body
            if self.args.use_fixed_body:
                if (self.args.fixed_bodies is None or len(self.args.fixed_bodies) == 0) and self.args.fixed_body_path is None:
                    raise ValueError("No fixed bodies specified")
                elif self.args.fixed_bodies is not None and len(self.args.fixed_bodies) > 0:
                    body = FIXED_BODY(fixed_body=self.args.fixed_bodies)
                elif self.args.fixed_body_path is not None:
                    body = FIXED_BODY(fixed_body_path=self.args.fixed_body_path)
                else:
                    raise ValueError("something surely wrong")
            else:
                body = BASIC_BODY(self.args)
            # brain
            if self.args.controller == 'DECENTRALIZED':
                brain = DECENTRALIZED(args=self.args)
            elif self.args.controller == 'CENTRALIZED':
                brain = CENTRALIZED(args=self.args)
            elif self.args.controller == 'TRANSFORMER':
                brain = TRANSFORMER(args=self.args)
            else:
                raise ValueError("Unknown brain type", self.args.controller)
            ind = INDIVIDUAL(body=body, brain=brain)
            if ind.is_valid():
                valid = True
        return ind

    def produce_offsprings(self):
        """Produce offspring from the current map."""
        # check if there are any individuals in the map
        if len(self) == 0:
            init_population = []
            while len(init_population) < self.args.nr_parents:
                init_population.append(self.get_random_individual())
            return init_population
        # choose nr_parents many random keys from the map. make sure that they are not None
        valid_keys = [ k for k in self.map.keys() if self.map[k] is not None ]
        nr_valid_keys = len(valid_keys) if len(valid_keys) < self.args.nr_parents else self.args.nr_parents
        random_keys_idx = np.random.choice(len(valid_keys), size=nr_valid_keys, replace=False)
        # produce offsprings
        offsprings = []
        for key_idx in random_keys_idx:
            key = valid_keys[key_idx]
            offsprings.append(self.map[key].produce_offspring())
        return offsprings

    def __iter__(self):
        """Iterate over the individuals. Use the expression 'for n in population'."""
        individuals = [ ind for ind in self.map.values() if ind is not None ]
        return iter(individuals)

    def __contains__(self, n):
        """Return True if n is a SoftBot in the population, False otherwise. Use the expression 'n in population'."""
        individuals = [ ind for ind in self.map.values() if ind is not None ]
        try:
            return n in individuals
        except TypeError:
            return False

    def __len__(self):
        """Return the number of individuals in the population. Use the expression 'len(population)'."""
        individuals = [ ind for ind in self.map.values() if ind is not None ]
        return len(individuals)

    def __getitem__(self, x, y):
        """Return individual n.  Use the expression 'population[n]'."""
        return self.map[(x,y)]

    def get_best_individual(self):
        """Return the best individual in the population."""
        individuals = [ ind for ind in self.map.values() if ind is not None ]
        return max(individuals, key=operator.attrgetter('fitness'))

    def get_best_fitness(self):
        """Return the best fitness in the population."""
        individuals = [ ind for ind in self.map.values() if ind is not None ]
        return max(individuals, key=operator.attrgetter('fitness')).fitness

    def update_map(self, population):
        """Update the map with the given population."""
        for ind in population:
            # determine the bins
            bin_existing_voxels, bin_active_voxels = self.determine_bins(ind)
            # update the map
            if self.map[(bin_existing_voxels, bin_active_voxels)] is None:
                self.map[(bin_existing_voxels, bin_active_voxels)] = ind
            else:
                if ind.fitness > self.map[(bin_existing_voxels, bin_active_voxels)].fitness:
                    self.map[(bin_existing_voxels, bin_active_voxels)] = ind

    def determine_bins(self, ind):
        """Calculate the bin indices for the given individual."""
        existing_voxels = ind.body.count_existing_voxels(ind.body.structure)
        active_voxels = ind.body.count_active_voxels(ind.body.structure)
        # determine the bin indices
        bin_existing_voxels = existing_voxels
        bin_active_voxels = active_voxels
        return bin_existing_voxels, bin_active_voxels

    def print_map(self):
        """Print some useful information about the map."""
        # print the best fitness in the map
        print("Best fitness in the map: ", self.get_best_individual().fitness)
        # print the occupancy of the map
        print("Occupancy of the map: ", len(self), "/", len(self.map))

    def get_fitnesses(self):
        """return a numpy array of fitnesses of the individuals in the map,
        with a mask to indicate which bins are not empty"""
        fitnesses = np.zeros((self.n_bins_existing_voxels, self.n_bins_active_voxels))
        for i in range(1, self.n_bins_existing_voxels+1):
            for j in range(1, self.n_bins_active_voxels+1):
                if self.map[(i,j)] is not None:
                    fitnesses[i-1,j-1] = self.map[(i,j)].fitness
                else:
                    fitnesses[i-1,j-1] = -9999
        # transpose the fitnesses
        fitnesses = fitnesses.transpose()
        # mask for non-empty bins
        mask = fitnesses != -9999
        return fitnesses, mask



        

