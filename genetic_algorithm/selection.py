import random


def tournament_selection(scored_population, tournament_size=3):
    """Sample a tournament and return a copy of its highest-scoring gait."""
    candidates = random.sample(scored_population, tournament_size)

    winner = max(candidates, key=lambda candidate: candidate[0])

    return winner[1][:]
