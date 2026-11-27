import numpy as np
import matplotlib.pyplot as plt


def plot_fitness(best_scores, average_scores):
    """Plot the best and mean fitness by generation."""
    generations = range(1, len(best_scores) + 1)

    plt.figure(figsize=(9, 5))

    plt.plot(generations, best_scores, label="Best fitness")
    plt.plot(generations, average_scores, label="Average fitness")

    plt.xlabel("Generation")
    plt.ylabel("Fitness score")
    plt.title("Genetic Algorithm Fitness Progress")
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_joint_angles(chromosome):
    """Plot the three L1 joint angles over one gait cycle."""
    from genetic_algorithm.chromosome import chromosome_to_gait

    gait = chromosome_to_gait(chromosome)

    frames = np.arange(len(gait))

    plt.figure(figsize=(9, 5))

    plt.plot(frames, gait[:, 0], label="L1 hip")
    plt.plot(frames, gait[:, 1], label="L1 thigh")
    plt.plot(frames, gait[:, 2], label="L1 knee")

    plt.xlabel("Frame")
    plt.ylabel("Joint angle (radians)")
    plt.title("L1 Joint Movement")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    from genetic_algorithm.population import create_population

    population = create_population()
    plot_joint_angles(population[0])

    # Example scores used to check the plotting routine.
    example_best = [10, 12, 15, 17, 19]
    example_average = [6, 8, 10, 12, 14]

    plot_fitness(example_best, example_average)
