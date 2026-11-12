
import random


def tournament_selection(scored_population, tournament_size=3):
    """
    Select a parent using tournament selection.

    scored_population: list of (fitness_score, chromosome) pairs
    tournament_size: number of candidates competing

    Returns a copy of the chromosome with the highest fitness
    among the randomly selected candidates.
    """

    # Randomly choose chromosomes to compete
    candidates = random.sample(scored_population, tournament_size)

    # Find the chromosome with the highest fitness score
    winner = max(candidates, key=lambda candidate: candidate[0])

    # Return a copy so the original chromosome is not modified
    return winner[1][:]
