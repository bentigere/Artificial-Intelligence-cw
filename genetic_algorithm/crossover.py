
import random


def crossover(parent1, parent2):
    # Crossover was chosen because it combines joint movements from two parent gaits
    # to create new walking patterns. This allows the GA to explore different
    # combinations of movements that may improve the spider's coordination and fitness.

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
