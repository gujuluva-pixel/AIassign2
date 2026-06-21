"""
experiment.py
Runs the full experiment required by the assignment:

  1. 30 runs of runGA(50, 0.7, 0.001)   -- mutation AND crossover
  2. 30 runs of runGA(50, 0.0, 0.001)   -- mutation ALONE (crossover off)

For each condition, records the generation at which the all-ones genome
was discovered, computes summary statistics, writes a CSV of the raw
per-run results, and produces a comparison plot of fitness-over-time
(averaged across the 30 runs, generation-by-generation) plus a bar
chart of generation-to-solution per run.
"""

import csv
import statistics
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ga import runGA, makePopulation, evaluateFitness, selectPair, crossover, mutate


NUM_RUNS = 30
POP_SIZE = 50
GENOME_LENGTH = 10
MUTATION_RATE = 0.001
MAX_GENERATIONS = 2000


def runGA_tracked(populationSize, crossoverRate, mutationRate,
                   genomeLength=10, maxGenerations=2000):
    """
    Same logic as ga.runGA, but silent (no printing) and also returns
    the full per-generation (avgFitness, bestFitness) history so we
    can plot learning curves, in addition to the generation at which
    the optimum was found.
    """
    population = makePopulation(populationSize, genomeLength)
    generation = 0
    history = []

    while True:
        averageFitness, bestFitness = evaluateFitness(population)
        history.append((averageFitness, bestFitness))

        if bestFitness == genomeLength or generation >= maxGenerations:
            return generation, history

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


def run_condition(label, crossoverRate):
    print(f"\n=== Running condition: {label} (pc={crossoverRate}) ===")
    generationsFound = []
    histories = []
    for run in range(1, NUM_RUNS + 1):
        gen, history = runGA_tracked(POP_SIZE, crossoverRate, MUTATION_RATE,
                                      GENOME_LENGTH, MAX_GENERATIONS)
        generationsFound.append(gen)
        histories.append(history)
        print(f"  Run {run:2d}: solution found at generation {gen}")
    return generationsFound, histories


def summarize(label, generationsFound):
    mean = statistics.mean(generationsFound)
    stdev = statistics.stdev(generationsFound) if len(generationsFound) > 1 else 0.0
    median = statistics.median(generationsFound)
    print(f"\n--- Summary: {label} ---")
    print(f"  Runs: {len(generationsFound)}")
    print(f"  Mean generation of discovery:   {mean:.2f}")
    print(f"  Std dev:                        {stdev:.2f}")
    print(f"  Median:                         {median}")
    print(f"  Min / Max:                      {min(generationsFound)} / {max(generationsFound)}")
    return {"mean": mean, "stdev": stdev, "median": median,
            "min": min(generationsFound), "max": max(generationsFound)}


def write_csv(filename, with_co, without_co):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "generations_to_solution_with_crossover_pc0.7",
                          "generations_to_solution_without_crossover_pc0.0"])
        for i in range(NUM_RUNS):
            writer.writerow([i + 1, with_co[i], without_co[i]])


def average_history(histories, maxLen):
    """
    Average best-fitness (and avg-fitness) across runs generation by
    generation. Runs that terminated early are padded by holding their
    final (avg, best) value constant for the remaining generations,
    which is a reasonable representation since best fitness can only
    stay at 10 once found.
    """
    avgCurve = []
    bestCurve = []
    for gIdx in range(maxLen):
        avgs = []
        bests = []
        for h in histories:
            if gIdx < len(h):
                a, b = h[gIdx]
            else:
                a, b = h[-1]
            avgs.append(a)
            bests.append(b)
        avgCurve.append(sum(avgs) / len(avgs))
        bestCurve.append(sum(bests) / len(bests))
    return avgCurve, bestCurve


def make_plots(hist_with, hist_without):
    maxLen = max(max(len(h) for h in hist_with), max(len(h) for h in hist_without))
    avg_w, best_w = average_history(hist_with, maxLen)
    avg_wo, best_wo = average_history(hist_without, maxLen)

    # Plot 1: averaged learning curves (best fitness) for both conditions
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(best_w, label="Best fitness (with crossover, pc=0.7)", color="#2E75B6", linewidth=2)
    ax.plot(best_wo, label="Best fitness (no crossover, pc=0.0)", color="#C0392B", linewidth=2)
    ax.plot(avg_w, label="Average fitness (with crossover)", color="#2E75B6", linestyle="--", alpha=0.6)
    ax.plot(avg_wo, label="Average fitness (no crossover)", color="#C0392B", linestyle="--", alpha=0.6)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (number of ones, max 10)")
    ax.set_title("GA Learning Curves: Crossover vs. No Crossover\n(averaged over 30 runs each)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("learning_curves.png", dpi=150)
    plt.close(fig)

    return maxLen


def make_bar_chart(gens_with, gens_without):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(1, NUM_RUNS + 1))
    width = 0.4
    ax.bar([i - width / 2 for i in x], gens_with, width=width,
           label="With crossover (pc=0.7)", color="#2E75B6")
    ax.bar([i + width / 2 for i in x], gens_without, width=width,
           label="No crossover (pc=0.0)", color="#C0392B")
    ax.set_xlabel("Run number")
    ax.set_ylabel("Generation solution found")
    ax.set_title("Generation at Which All-Ones Genome Was Found, per Run")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("generations_per_run.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    gens_with, hist_with = run_condition("WITH crossover", 0.7)
    gens_without, hist_without = run_condition("WITHOUT crossover", 0.0)

    stats_with = summarize("With crossover (pc=0.7)", gens_with)
    stats_without = summarize("Without crossover (pc=0.0)", gens_without)

    write_csv("results.csv", gens_with, gens_without)
    make_plots(hist_with, hist_without)
    make_bar_chart(gens_with, gens_without)

    print("\nWrote results.csv, learning_curves.png, generations_per_run.png")
