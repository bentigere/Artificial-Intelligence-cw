import random


def crossover(parent1, parent2):
    """Build a child by choosing each joint angle from either parent."""
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length")

    child = []

    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child.append(gene1)
        else:
            child.append(gene2)

    return child
