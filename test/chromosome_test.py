
import numpy as np

from genetic_algorithm.chromosome import (
    create_chromosome,
    chromosome_to_gait,
    NUM_FRAMES,
    NUM_GENES,
    LOWER_LIMITS,
    UPPER_LIMITS
)


def test_chromosome_length():
    """A chromosome must contain 7200 genes."""
    chromosome = create_chromosome()

    assert chromosome.shape == (NUM_GENES,)


def test_gait_shape():
    """The chromosome must convert into 300 frames."""
    chromosome = create_chromosome()
    gait = chromosome_to_gait(chromosome)

    assert gait.shape == (NUM_FRAMES, 24)


def test_joint_limits():
    """Every generated angle must respect motor limits."""
    chromosome = create_chromosome()
    gait = chromosome_to_gait(chromosome)

    assert np.all(gait >= LOWER_LIMITS)
    assert np.all(gait <= UPPER_LIMITS)


def test_invalid_chromosome():
    """Reject a chromosome with the wrong number of genes."""
    try:
        chromosome_to_gait(np.zeros(100))
    except ValueError:
        return

    raise AssertionError("Invalid chromosome was accepted")
