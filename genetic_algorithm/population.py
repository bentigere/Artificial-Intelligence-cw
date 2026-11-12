
import numpy as np

from chromosome import (
    create_chromosome,
    chromosome_to_gait,
    NUM_FRAMES,
    ANGLES_PER_FRAME
)

# Number of candidate walking gaits
POPULATION_SIZE = 20


def create_population(size=POPULATION_SIZE):
    """
    Create a population of candidate spider gaits.

    Each chromosome contains 7200 joint angles,
    representing 300 frames with 24 angles per frame.
    """

    rng = np.random.default_rng(42)

    population = []

    # Generate one chromosome for each individual
    for _ in range(size):
        chromosome = create_chromosome(rng)
        population.append(chromosome)

    return np.array(population)


# Test population initialization
if __name__ == "__main__":

    population = create_population()

    print("Population size:", len(population))
    print("Genes per chromosome:", len(population[0]))
    print("Population shape:", population.shape)

    # Check the first chromosome
    first_gait = chromosome_to_gait(population[0])

    print("First gait shape:", first_gait.shape)

    # Verify that different individuals were generated
    same = np.array_equal(population[0], population[1])

    print("First two chromosomes identical:", same)
