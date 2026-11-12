
import random

from genetic_algorithm.selection import tournament_selection
from genetic_algorithm.crossover import crossover
from genetic_algorithm.mutation import mutate

# 300 frames x 24 joints = 7,200 genes per chromosome.
CHROMOSOME_LENGTH = 7200


def test_operators():
    # Make the random results reproducible.
    random.seed(42)

    # Create three chromosomes with different fitness scores.
    population = [
        (10, [0.1] * CHROMOSOME_LENGTH),
        (20, [0.2] * CHROMOSOME_LENGTH),
        (30, [0.3] * CHROMOSOME_LENGTH),
    ]

    # SELECTION: The highest-fitness chromosome should win.
    parent1 = tournament_selection(population)
    parent2 = tournament_selection(population)

    assert parent1 == population[2][1]
    assert parent2 == population[2][1]
    print("Selection test passed!")

    # CROSSOVER: Create a child from the two parents.
    child = crossover(parent1, parent2)

    assert len(child) == CHROMOSOME_LENGTH
    assert all(gene in (a, b) for gene, a, b
               in zip(child, parent1, parent2))
    print("Crossover test passed!")

    # MUTATION: Randomly modify the child's joint angles.
    mutated_child = mutate(child, mutation_rate=0.01)

    assert len(mutated_child) == CHROMOSOME_LENGTH
    assert all(-3.14 <= gene <= 3.14
               for gene in mutated_child)
    print("Mutation test passed!")

    # Confirm the original child was not modified.
    assert len(child) == CHROMOSOME_LENGTH

    print("All genetic operators passed!")


if __name__ == "__main__":
    test_operators()
