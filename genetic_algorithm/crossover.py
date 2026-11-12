
import random


def crossover(parent1, parent2):
    """
    Combine two parent chromosomes using uniform crossover.

    Each gene has a 50% chance of coming from either parent.

    Args:
        parent1: First chromosome (list of joint angles).
        parent2: Second chromosome (list of joint angles).

    Returns:
        A new chromosome containing genes from both parents.
    """

    # Both parents must have the same number of genes.
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length")

    # Create a new child chromosome.
    child = []

    # Examine corresponding genes from both parents.
    for gene1, gene2 in zip(parent1, parent2):

        # Randomly select which parent's gene to inherit.
        if random.random() < 0.5:
            child.append(gene1)
        else:
            child.append(gene2)

    # Return the new chromosome without changing the parents.
    return child
