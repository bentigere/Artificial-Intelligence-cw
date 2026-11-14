
import numpy as np
from pathlib import Path

from genetic_algorithm.population import create_population
from genetic_algorithm.chromosome import chromosome_to_gait


def export_gait(chromosome, filename):
    """Convert a chromosome into a CSV for Webots."""

    # Reshape 7200 genes into 300 frames x 24 angles
    gait = chromosome_to_gait(chromosome)

    # Create the output folder if needed
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Save each frame as one row of 24 angles
    np.savetxt(filename, gait, delimiter=",", fmt="%.8f")

    print(f"Saved gait to: {filename}")
    print(f"Gait shape: {gait.shape}")


if __name__ == "__main__":

    # Use one smooth candidate for initial Webots testing
    population = create_population()
    test_chromosome = population[0]

    # Save it in the project's data folder
    output = Path("data/spider_gait.csv")

    export_gait(test_chromosome, output)
