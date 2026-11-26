
import random


def tournament_selection(scored_population, tournament_size=3):
    # Tournament selection was chosen because it is simple and gives
    # better-performing spider gaits a higher chance of being selected.
    # It also maintains population diversity and helps avoid premature convergence.
    # A tournament size of 3 balances selecting good solutions with exploring others.

    # Randomly choose chromosomes to compete
    candidates = random.sample(scored_population, tournament_size)

    # Find the chromosome with the highest fitness score
    winner = max(candidates, key=lambda candidate: candidate[0])

    # Return a copy so the original chromosome is not modified
    return winner[1][:]
