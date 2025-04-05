import os
import dill as pickle
import matplotlib.pyplot as plt
import pygame
import config as cfg
from neural_network import NeuralNetwork 
if not pygame.mixer.get_init():
    pygame.mixer.init()

sounds_loaded = False
sound_jump = None
sound_die = None

try:
    sound_jump = pygame.mixer.Sound(cfg.JUMP_SOUND_PATH)
    sound_die = pygame.mixer.Sound(cfg.DIE_SOUND_PATH)
    sounds_loaded = True
    print("Sounds loaded successfully.")
except pygame.error as e:
    print(f"Warning: Could not load one or more sounds: {e}")
    sounds_loaded = False
except FileNotFoundError as e:
    print(f"Warning: Sound file not found: {e}")
    sounds_loaded = False

def play_sound(sound_object):
    if sounds_loaded and sound_object:
        sound_object.play()

def save_best_brain(brain, filename=cfg.BRAIN_FILENAME):
    try:
        with open(filename, "wb") as f:
            pickle.dump(brain, f)
    except Exception as e:
        print(f"Error saving brain to {filename}: {e}")

def load_best_brain(filename=cfg.BRAIN_FILENAME):
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                print(f"Loading best brain from {filename}")
                # When loading, pickle needs the class definition available
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading brain from {filename}: {e}. Starting fresh.")
            return None
    else:
        print("No saved brain found. Starting fresh.")
        return None

def plot_scores(scores, filename=cfg.PLOT_FILENAME):
    plt.figure(figsize=(10, 5))
    plt.plot(scores, marker='.', linestyle='-')
    plt.title("Best Dino Score per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Highest Score")
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(filename)
    except Exception as e:
        print(f"Could not save plot: {e}")
    plt.close()