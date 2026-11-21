
import random
import numpy as np

from genetic_algorithm.population import (
    create_population,
    MATLAB_STANDING_POSE,
)

from genetic_algorithm.updated_mutation import updated_mutate

from genetic_algorithm.chromosome import (
    LOWER_LIMITS,
    UPPER_LIMITS,
)


def test_mutation_shape():
    population = create_population(
        reference_pose=MATLAB_STANDING_POSE
    )

    child = updated_mutate(population[0])

    assert len(child) == 7200


def test_mutation_joint_limits():
    population = create_population(
        reference_pose=MATLAB_STANDING_POSE
    )

    child = np.array(
        updated_mutate(population[0])
    ).reshape(300, 24)

    assert np.all(child >= LOWER_LIMITS)
    assert np.all(child <= UPPER_LIMITS)


def test_mutation_changes_chromosome():
    random.seed(42)

    population = create_population(
        reference_pose=MATLAB_STANDING_POSE
    )

    original = population[0]

    mutated = np.array(
        updated_mutate(
            original,
            mutation_rate=1.0,
            mutation_strength=0.05
        )
    )

    assert not np.array_equal(original, mutated)


def test_mutation_preserves_smoothness():
    random.seed(42)

    population = create_population(
        reference_pose=MATLAB_STANDING_POSE
    )

    mutated = np.array(
        updated_mutate(
            population[0],
            mutation_rate=1.0,
            mutation_strength=0.05
        )
    ).reshape(300, 24)

    changes = np.abs(np.diff(mutated, axis=0))

    loop_change = np.abs(
        mutated[-1] - mutated[0]
    )

    assert np.max(changes) < 0.1
    assert np.max(loop_change) < 0.1


def test_invalid_mutation_rate():
    population = create_population(
        reference_pose=MATLAB_STANDING_POSE
    )

    try:
        updated_mutate(
            population[0],
            mutation_rate=1.5
        )
    except ValueError:
        return

    raise AssertionError("Expected ValueError")
