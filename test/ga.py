import math
import random
from pathlib import Path
from gait import generatep_gait, save_csv

# Parameter order: yaw amplitude, femur amplitude, tibia amplitude,
# femur offset, tibia offset, tibia phase (all radians).
BOUNDS = [
    (0.10, 0.60),
    (0.10, 0.50),
    (0.10, 0.60),
    (-0.50, 0.20),
    (-1.00, -0.20),
    (0.20, 2.50),
]
POPULATION_SIZE = 50
GENERATIONS = 60
MUTATION_RATE = 0.20
ELITE_COUNT = 2


def create_chromosome():
    return [random.uniform(low, high) for low, high in BOUNDS]


def gait_fitness(chromosome):
    """PROTOTYPE heuristic: smoothness + movement + joint-angle bounds.

    This is NOT a physical walking/stability score. Replace or extend it
    using foot contacts, support, collision and displacement in simulation.
    """
    poses = generate_gait(chromosome)
    # Reward modest swing and knee movement; penalize abrupt joint changes.
    motion = sum(abs(poses[i][0] - poses[i - 1][0]) for i in range(1, len(poses)))
    motion += sum(abs(poses[i][2] - poses[i - 1][2]) for i in range(1, len(poses)))
    jerk = sum(
        (poses[i][j] - 2 * poses[i - 1][j] + poses[i - 2][j]) ** 2
        for i in range(2, len(poses))
        for j in range(24)
    )
    return motion / (1.0 + 10.0 * jerk)


def tournament_selection(scored, size=3):
    return max(random.sample(scored, size), key=lambda pair: pair[0])[1][:]


def crossover(parent1, parent2):
    return [a if random.random() < 0.5 else b for a, b in zip(parent1, parent2)]


def mutate(chromosome):
    child = chromosome[:]
    for i, (low, high) in enumerate(BOUNDS):
        if random.random() < MUTATION_RATE:
            child[i] += random.gauss(0, 0.10 * (high - low))
            child[i] = max(low, min(high, child[i]))
    return child


def run_ga():
    population = [create_chromosome() for _ in range(POPULATION_SIZE)]
    best_score = float("-inf")
    best = None
    for generation in range(GENERATIONS):
        scored = sorted(((gait_fitness(c), c) for c in population), reverse=True)
        if scored[0][0] > best_score:
            best_score, best = scored[0][0], scored[0][1][:]
        if generation % 10 == 0 or generation == GENERATIONS - 1:
            print(
                f"Generation {generation + 1}: best prototype score = {best_score:.4f}"
            )
        next_population = [c[:] for _, c in scored[:ELITE_COUNT]]
        while len(next_population) < POPULATION_SIZE:
            p1 = tournament_selection(scored)
            p2 = tournament_selection(scored)
            next_population.append(mutate(crossover(p1, p2)))
        population = next_population
    return best, best_score


if __name__ == "__main__":
    random.seed(42)  # Reproducible test run
    best, score = run_ga()
    poses = generate_gait(best)
    # change output path
    output = Path(__file__).resolve().parent.parent / "data" / "spider_gait.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_csv(poses, output)
    print("Best gait parameters:", [round(x, 4) for x in best])
    print("Prototype fitness:", round(score, 4))
    print(f"Saved {len(poses)} poses, {len(poses[0])} angles each, to {output}")
