import numpy as np
from pathlib import Path

from genetic_algorithm.population import create_population
from genetic_algorithm.chromosome import chromosome_to_gait


def export_gait(chromosome, filename):
    """Write a headerless Webots CSV with one row per gait frame."""
    gait = chromosome_to_gait(chromosome)

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(filename, gait, delimiter=",", fmt="%.8f")

    print(f"Saved gait to: {filename}")
    print(f"Gait shape: {gait.shape}")


if __name__ == "__main__":

    # Export a seed gait for a Webots smoke test.
    population = create_population()
    test_chromosome = population[0]

    output = Path("data/spider_gait.csv")

    export_gait(test_chromosome, output)
