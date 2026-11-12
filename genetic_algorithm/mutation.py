
import random


def mutate(chromosome, mutation_rate=0.01,
           mutation_strength=0.1,
           min_angle=-3.14, max_angle=3.14):
    """
    Apply random mutations to a chromosome.

    Args:
        chromosome: List of joint angles in radians.
        mutation_rate: Probability of changing each gene.
        mutation_strength: Standard deviation of the change.
        min_angle: Minimum permitted angle.
        max_angle: Maximum permitted angle.

    Returns:
        A new chromosome with possible mutations.
    """

    # Make a copy to protect the original chromosome.
    mutated = chromosome[:]

    # Go through every gene in the chromosome.
    for i in range(len(mutated)):

        # Decide whether this gene should mutate.
        if random.random() < mutation_rate:

            # Generate a small random change.
            change = random.gauss(0, mutation_strength)

            # Apply the change to the joint angle.
            new_angle = mutated[i] + change

            # Keep the angle within the specified limits.
            new_angle = max(min_angle, min(max_angle, new_angle))

            # Store the mutated angle.
            mutated[i] = new_angle

    # Return the new chromosome.
    return mutated
