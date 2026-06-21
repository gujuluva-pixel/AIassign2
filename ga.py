"""
ga.py
A simple genetic algorithm (GA) for maximizing the "ones counting" fitness
function f(x) = number of 1 bits in x, where x is a bit string (genome)
of length 10.

Implements, per the assignment spec:
    randomGenome(length)
    makePopulation(size, length)
    fitness(genome)
    evaluateFitness(population)
    selectPair(population)
    crossover(genome1, genome2)
    mutate(genome, mutationRate)
    runGA(populationSize, crossoverRate, mutationRate)

A genome is represented as a list of ints (0/1), e.g. [1,0,1,1,0,...].
"""

import random


def randomGenome(length):
    """Return a random genome (list of 0/1) of the given length."""
    return [random.randint(0, 1) for _ in range(length)]


def makePopulation(size, length):
    """Return a list of `size` randomly created genomes, each of `length` bits."""
    return [randomGenome(length) for _ in range(size)]


def fitness(genome):
    """Fitness = number of ones in the genome."""
    return sum(genome)


def evaluateFitness(population):
    """Return (averageFitness, bestFitness) for the population."""
    fitnesses = [fitness(g) for g in population]
    averageFitness = sum(fitnesses) / len(fitnesses)
    bestFitness = max(fitnesses)
    return averageFitness, bestFitness


def selectPair(population):
    """
    Select two genomes from the population using fitness-proportionate
    (roulette-wheel) selection, with replacement. Returns a tuple of two
    genomes (copies are NOT made here; caller should copy if needed).
    """
    fitnesses = [fitness(g) for g in population]
    totalFitness = sum(fitnesses)

    # Degenerate case: every genome is all zeros -> uniform random pick
    if totalFitness == 0:
        return random.choice(population), random.choice(population)

    def rouletteWheelPick():
        spin = random.uniform(0, totalFitness)
        running = 0.0
        for genome, f in zip(population, fitnesses):
            running += f
            if running >= spin:
                return genome
        return population[-1]  # fallback for floating point edge case

    parent1 = rouletteWheelPick()
    parent2 = rouletteWheelPick()
    return parent1, parent2


def crossover(genome1, genome2):
    """
    Single-point crossover. Picks a random crossover point in
    [1, length-1] and swaps the tails of the two genomes, returning
    two new (child) genomes.
    """
    length = len(genome1)
    point = random.randint(1, length - 1)
    child1 = genome1[:point] + genome2[point:]
    child2 = genome2[:point] + genome1[point:]
    return child1, child2


def mutate(genome, mutationRate):
    """
    Bitwise mutation: each bit is independently flipped with
    probability mutationRate. Returns a NEW genome (does not
    modify the input in place).
    """
    return [1 - bit if random.random() < mutationRate else bit for bit in genome]


def runGA(populationSize, crossoverRate, mutationRate,
          genomeLength=10, maxGenerations=2000, verbose=True):
    """
    Main GA loop.

    Repeatedly evaluates the population, prints average/best fitness,
    and (unless the optimum 1111111111 has been found, or maxGenerations
    is reached) breeds a new generation via fitness-proportionate
    selection, single-point crossover (applied with probability
    crossoverRate), and bitwise mutation (applied with probability
    mutationRate per bit).

    Returns the generation number at which the all-ones genome was
    first found (or maxGenerations if it was never found).
    """
    population = makePopulation(populationSize, genomeLength)
    generation = 0

    while True:
        averageFitness, bestFitness = evaluateFitness(population)
        if verbose:
            print(f"Generation {generation}: average fitness "
                  f"{averageFitness:.2f}, best fitness {bestFitness:.2f}")

        if bestFitness == genomeLength or generation >= maxGenerations:
            return generation

        newPopulation = []
        while len(newPopulation) < populationSize:
            parent1, parent2 = selectPair(population)

            if random.random() < crossoverRate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            child1 = mutate(child1, mutationRate)
            child2 = mutate(child2, mutationRate)

            newPopulation.append(child1)
            if len(newPopulation) < populationSize:
                newPopulation.append(child2)

        population = newPopulation
        generation += 1


if __name__ == "__main__":
    # Demonstration run matching the assignment's example output format.
    print(">>> runGA(50, 0.0, 0.001)")
    print("Population size: 50")
    print("Genome length: 10")
    random.seed()  # nondeterministic demo run
    runGA(50, 0.7, 0.001)
