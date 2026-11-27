import random


def mutate(
    chromosome,
    mutation_rate=0.01,
    mutation_strength=0.1,
    min_angle=-3.14,
    max_angle=3.14,
):
    """Perturb individual angles with Gaussian noise and clip changed values.

    Args:
        chromosome: List of joint angles in radians.
        mutation_rate: Probability of changing each gene.
        mutation_strength: Standard deviation of the change.
        min_angle: Minimum permitted angle.
        max_angle: Maximum permitted angle.

    Returns:
        A new chromosome with possible mutations.
    """

    mutated = chromosome[:]

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            change = random.gauss(0, mutation_strength)
            new_angle = mutated[i] + change
            new_angle = max(min_angle, min(max_angle, new_angle))
            mutated[i] = new_angle

    return mutated
