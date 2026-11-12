
from spider import calculate_spider_feet


def calculate_fitness(chromosome):
    # Calculate all 8 foot positions
    feet = calculate_spider_feet(chromosome)

    # Extract the height (Z coordinate) of each foot
    foot_heights = []

    for position in feet.values():
        foot_heights.append(position[2])

    # Find the lowest foot
    ground_level = min(foot_heights)

    # Calculate total difference from that level
    total_error = 0

    for height in foot_heights:
        difference = abs(height - ground_level)
        total_error += difference

    # Lower error means better foot alignment
    fitness = 1 / (1 + total_error)

    return fitness


# Test the fitness function
if __name__ == "__main__":
    # All joint angles are zero
    chromosome = [0.0] * 24

    score = calculate_fitness(chromosome)

    print("Spider fitness score:", score)
