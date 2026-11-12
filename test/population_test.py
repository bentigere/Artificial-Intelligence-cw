
import numpy as np

from genetic_algorithm.population import create_population
from genetic_algorithm.chromosome import (
    chromosome_to_gait,
    LOWER_LIMITS,
    UPPER_LIMITS
)


def test_population_shape():
    """Population must contain 20 chromosomes."""
    population = create_population()

    assert population.shape == (20, 7200)


def test_population_diversity():
    """The first two chromosomes should be different."""
    population = create_population()

    assert not np.array_equal(
        population[0],
        population[1]
    )


def test_population_joint_limits():
    """Every angle in every chromosome must be valid."""
    population = create_population()

    for chromosome in population:
        gait = chromosome_to_gait(chromosome)

        assert np.all(gait >= LOWER_LIMITS)
        assert np.all(gait <= UPPER_LIMITS)


def test_population_smoothness():
    """Neighbouring frames should change gradually."""
    population = create_population()

    for chromosome in population:
        gait = chromosome_to_gait(chromosome)

        # Include the transition back to frame zero
        changes = np.diff(
            np.vstack([gait, gait[0]]),
            axis=0
        )

        assert np.max(np.abs(changes)) < 0.05
