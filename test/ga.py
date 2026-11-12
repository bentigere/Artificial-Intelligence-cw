
import math
import random
from pathlib import Path
from gait import generate_gait, save_csv

# Parameter order: yaw amplitude, femur amplitude, tibia amplitude,
# femur offset, tibia offset, tibia phase (all radians).
BOUNDS = [(0.10, 0.60), (0.10, 0.50), (0.10, 0.60),
          (-0.50, 0.20), (-1.00, -0.20), (0.20, 2.50)]
POPULATION_SIZE = 50
GENERATIONS = 60
MUTATION_RATE = 0.20
ELITE_COUNT = 2


def create_chromosome():
    return [random.uniform(low, high) for low, high in BOUNDS]



def gait_fitness(chromosome):
    """Evaluate a candidate gait using heuristic penalties."""

    poses = generate_gait(chromosome)
    n = len(poses)

    # 1. Movement reward: encourage controlled yaw movement.
    movement = 0.0
    for pose in poses:
        movement += sum(abs(pose[j]) for j in range(0, 24, 3))
    movement /= n * 8

    # 2. Smoothness penalty: discourage sudden joint changes.
    # Include the wraparound between the last and first frames.
    smoothness = 0.0
    for i in range(n):
        previous = poses[(i - 1) % n]
        current = poses[i]
        following = poses[(i + 1) % n]

        for j in range(24):
            acceleration = (
                following[j] - 2 * current[j] + previous[j]
            )
            smoothness += acceleration ** 2

    smoothness /= n * 24

    # 3. Joint-limit penalty: penalise angles outside
    # provisional joint limits (radians).
    joint_limits = [
        (-0.8, 0.8),    # yaw
        (-0.7, 0.7),    # femur
        (-1.4, 0.2)     # tibia
    ]

    joint_penalty = 0.0

    for pose in poses:
        for j, angle in enumerate(pose):
            low, high = joint_limits[j % 3]

            if angle < low:
                joint_penalty += (low - angle) ** 2
            elif angle > high:
                joint_penalty += (angle - high) ** 2

    joint_penalty /= n * 24

    # 4. Coordination penalty: opposite-phase legs
    # should move in opposite yaw directions.
    coordination_penalty = 0.0

    for pose in poses:
        for leg in range(0, 8, 2):
            yaw_a = pose[leg * 3]
            yaw_b = pose[(leg + 1) * 3]
            coordination_penalty += (yaw_a + yaw_b) ** 2

    coordination_penalty /= n * 4

    # Final weighted fitness.
    fitness = (
        5.0 * movement
        - 10.0 * smoothness
        - 5.0 * joint_penalty
        - 2.0 * coordination_penalty
    )

    return fitness



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
    best_score = float('-inf')
    best = None
    for generation in range(GENERATIONS):
        scored = sorted(((gait_fitness(c), c) for c in population), reverse=True)
        if scored[0][0] > best_score:
            best_score, best = scored[0][0], scored[0][1][:]
        if generation % 10 == 0 or generation == GENERATIONS - 1:
            print(f'Generation {generation + 1}: best prototype score = {best_score:.4f}')
        next_population = [c[:] for _, c in scored[:ELITE_COUNT]]
        while len(next_population) < POPULATION_SIZE:
            p1 = tournament_selection(scored)
            p2 = tournament_selection(scored)
            next_population.append(mutate(crossover(p1, p2)))
        population = next_population
    return best, best_score


if __name__ == '__main__':
    random.seed(42)  # Reproducible test run
    best, score = run_ga()
    poses = generate_gait(best)
    output = Path(__file__).resolve().parent.parent / 'data' / 'spider_gait.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    save_csv(poses, output)
    print('Best gait parameters:', [round(x, 4) for x in best])
    print('Prototype fitness:', round(score, 4))
    print(f'Saved {len(poses)} poses, {len(poses[0])} angles each, to {output}')

