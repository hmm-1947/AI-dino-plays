import random
import numpy as np
import config as cfg
from entities import Dino
from neural_network import NeuralNetwork

def select_parent_tournament(dinos, tournament_size=cfg.TOURNAMENT_SIZE):
    if not dinos: return None
    actual_tournament_size = min(len(dinos), tournament_size)
    if actual_tournament_size <= 0: return None
    tournament = random.sample(dinos, actual_tournament_size)
    winner = max(tournament, key=lambda d: d.score)
    return winner

def evolve_population(old_dinos, mutation_rate=cfg.MUTATION_RATE, mutation_amount=cfg.MUTATION_AMOUNT):
    if not old_dinos: return []
    sorted_dinos = sorted(old_dinos, key=lambda d: d.score, reverse=True)
    best_score = sorted_dinos[0].score if sorted_dinos else 0
    print(f"Gen Best Score: {int(best_score)}")
    new_dinos = []
    elite_count = max(1, int(cfg.POPULATION_SIZE * cfg.ELITISM_PERCENT))
    for i in range(elite_count):
        if i < len(sorted_dinos):
            child = Dino(cfg.GROUND_Y)
            child.brain = sorted_dinos[i].brain.clone()
            new_dinos.append(child)

    while len(new_dinos) < cfg.POPULATION_SIZE:
        parent1 = select_parent_tournament(sorted_dinos)
        parent2 = select_parent_tournament(sorted_dinos)
        while parent2 == parent1 and len(sorted_dinos) > 1:
             parent2 = select_parent_tournament(sorted_dinos)
        if not parent1 or not parent2:
             print("Warning: Parent selection failed. Creating random offspring.")
             child = Dino(cfg.GROUND_Y)
        else:
             child = Dino(cfg.GROUND_Y)
             NeuralNetwork.crossover(parent1.brain, parent2.brain, child.brain)
             child.brain.mutate(rate=mutation_rate, amount=mutation_amount)
        new_dinos.append(child)
    return new_dinos[:cfg.POPULATION_SIZE]