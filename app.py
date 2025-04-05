import pygame
import random
import sys
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
import os
import time
import math # For sigmoid/tanh calculation maybe

# --- Pygame Setup ---
pygame.init()
pygame.mixer.init() # Initialize the mixer for sound

# --- Game Settings ---
WIDTH, HEIGHT = 900, 450 # Slightly wider screen
FPS = 60
GRAVITY = 0.7
JUMP_STRENGTH = -16 # Adjusted jump strength
DUCK_Y_OFFSET = 25 # How much the dino lowers when ducking
POPULATION_SIZE = 30 # Increased population
GROUND_Y = HEIGHT - 70 # Adjusted ground level slightly

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
GREY = (100, 100, 100)
SKY_BLUE = (135, 206, 250) # Background color

# --- Paths to Assets ---
ASSET_FOLDER = "assets"
DINO_IMG_PATH = os.path.join(ASSET_FOLDER, "dino.png")
DINO_DUCK_IMG_PATH = os.path.join(ASSET_FOLDER, "dino_duck.png")
BASE_CACTUS_IMG_PATH = os.path.join(ASSET_FOLDER, "cactus.png")
PTERODACTYL_IMG_PATH = os.path.join(ASSET_FOLDER, "pterodactyl.png")
JUMP_SOUND_PATH = os.path.join(ASSET_FOLDER, "jump.wav")
DIE_SOUND_PATH = os.path.join(ASSET_FOLDER, "die.wav")
POINT_SOUND_PATH = os.path.join(ASSET_FOLDER, "point.wav") # Optional

# --- Cactus Types (Using Scaling) ---
CACTUS_TYPES = [
    {'scale': (25, 50), 'count': 1},  # Small single
    {'scale': (40, 70), 'count': 1},  # Large single
    {'scale': (50, 50), 'count': 2},  # Double small
    {'scale': (75, 50), 'count': 3},  # Triple small
]

# --- Control Keys ---
KEY_PAUSE = pygame.K_p
KEY_SPEED_UP = pygame.K_PLUS # Or K_EQUALS if PLUS doesn't work
KEY_SLOW_DOWN = pygame.K_MINUS
KEY_MANUAL_JUMP = pygame.K_SPACE # For testing
KEY_MANUAL_DUCK = pygame.K_DOWN    # For testing

# --- Simulation Speed Control ---
simulation_speed_factor = 1.0 # 1.0 = normal, >1 = faster, <1 = slower
min_sim_speed = 0.25
max_sim_speed = 4.0

# --- Pygame Display & Clock ---
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Enhanced AI Chrome Dino")
clock = pygame.time.Clock()
FONT_SMALL = pygame.font.SysFont(None, 24)
FONT_MEDIUM = pygame.font.SysFont(None, 30)

# --- Load Sounds ---
try:
    sound_jump = pygame.mixer.Sound(JUMP_SOUND_PATH)
    sound_die = pygame.mixer.Sound(DIE_SOUND_PATH)
    # sound_point = pygame.mixer.Sound(POINT_SOUND_PATH) # Uncomment if you have it
    sounds_loaded = True
except pygame.error as e:
    print(f"Warning: Could not load one or more sounds: {e}")
    sounds_loaded = False

def play_sound(sound):
    if sounds_loaded:
        sound.play()

# --- Neural Network Class ---
class NeuralNetwork:
    # Inputs: dist1, height1, width1, dist2, speed, dino_vy
    def __init__(self, input_size=6, hidden_size=8, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Xavier/Glorot Initialization (good for Tanh/Sigmoid)
        limit_ih = np.sqrt(6. / (self.input_size + self.hidden_size))
        self.weights_input_hidden = np.random.uniform(-limit_ih, limit_ih, (self.input_size, self.hidden_size))
        self.bias_hidden = np.zeros(self.hidden_size)

        limit_ho = np.sqrt(6. / (self.hidden_size + self.output_size))
        self.weights_hidden_output = np.random.uniform(-limit_ho, limit_ho, (self.hidden_size, self.output_size))
        self.bias_output = np.zeros(self.output_size)

    def forward(self, inputs):
        inputs = np.array(inputs).flatten()
        if inputs.shape[0] != self.input_size:
            # Pad or truncate if size mismatch (basic handling)
            inputs = np.resize(inputs, self.input_size)

        hidden_raw = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_activated = self.tanh(hidden_raw) # Using Tanh activation

        output_raw = np.dot(hidden_activated, self.weights_hidden_output) + self.bias_output
        output_activated = self.sigmoid(output_raw) # Sigmoid for output [0, 1]

        return output_activated[0] # Return the single output value

    # Activation Functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    def tanh(self, x):
        return np.tanh(np.clip(x, -15, 15)) # Tanh is often good for hidden layers

    # --- Genetic Algorithm Methods ---
    def clone(self):
        clone = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        clone.weights_input_hidden = np.copy(self.weights_input_hidden)
        clone.bias_hidden = np.copy(self.bias_hidden)
        clone.weights_hidden_output = np.copy(self.weights_hidden_output)
        clone.bias_output = np.copy(self.bias_output)
        return clone

    def mutate(self, rate=0.1, amount=0.2):
        def mutate_array(arr):
            mutation_mask = np.random.rand(*arr.shape) < rate
            random_mutation = (np.random.rand(*arr.shape) - 0.5) * amount * 2 # Centered mutation
            arr += mutation_mask * random_mutation
        mutate_array(self.weights_input_hidden)
        mutate_array(self.bias_hidden)
        mutate_array(self.weights_hidden_output)
        mutate_array(self.bias_output)

    @staticmethod
    def crossover(brain1, brain2, child_brain):
        """ Performs average crossover between two parent brains into child_brain """
        # Crossover Weights Input -> Hidden
        mask = np.random.rand(*brain1.weights_input_hidden.shape) > 0.5
        child_brain.weights_input_hidden = np.where(mask, brain1.weights_input_hidden, brain2.weights_input_hidden)
        # Crossover Bias Hidden
        mask = np.random.rand(*brain1.bias_hidden.shape) > 0.5
        child_brain.bias_hidden = np.where(mask, brain1.bias_hidden, brain2.bias_hidden)
        # Crossover Weights Hidden -> Output
        mask = np.random.rand(*brain1.weights_hidden_output.shape) > 0.5
        child_brain.weights_hidden_output = np.where(mask, brain1.weights_hidden_output, brain2.weights_hidden_output)
         # Crossover Bias Output
        mask = np.random.rand(*brain1.bias_output.shape) > 0.5
        child_brain.bias_output = np.where(mask, brain1.bias_output, brain2.bias_output)


# --- Base Obstacle Class ---
class Obstacle:
    def __init__(self, x_pos, speed):
        self.x = x_pos
        self.speed = speed
        self.image = None
        self.width = 0
        self.height = 0
        self.passed = False # To potentially award points only once

    def update(self):
        self.x -= self.speed * simulation_speed_factor

    def draw(self, surface):
        if self.image:
            surface.blit(self.image, (int(self.x), int(self.y)))
            # Optional: Draw hitbox
            # pygame.draw.rect(surface, RED, self.get_rect(), 1)

    def off_screen(self):
        return self.x + self.width < 0

    def get_rect(self):
        # Default simple rect, subclasses might refine padding
        return pygame.Rect(self.x, self.y, self.width, self.height)

# --- Cactus Class (inherits from Obstacle) ---
class Cactus(Obstacle):
    def __init__(self, x_pos, ground_y, cactus_info, base_image_path, speed):
        super().__init__(x_pos, speed)
        self.image_loaded = False
        self.width = 30 # Default
        self.height = 60 # Default

        try:
            if 'scale' in cactus_info and base_image_path and os.path.exists(base_image_path):
                base_image = pygame.image.load(base_image_path).convert_alpha()
                self.width, self.height = cactus_info['scale']
                self.image = pygame.transform.smoothscale(base_image, (self.width, self.height))
                self.image_loaded = True
            else: self._create_fallback()
        except Exception as e: print(f"Error loading/scaling cactus: {e}"); self._create_fallback()

        self.y = ground_y - self.height # Top y coordinate

    def _create_fallback(self):
        self.width = self.width if hasattr(self, 'width') else 30
        self.height = self.height if hasattr(self, 'height') else 60
        self.image = pygame.Surface((self.width, self.height)); self.image.fill(GREEN)
        self.image_loaded = True

    def get_rect(self):
        padding = 3
        rect_w = max(1, self.width - 2 * padding)
        rect_h = max(1, self.height - 2 * padding)
        return pygame.Rect(self.x + padding, self.y + padding, rect_w, rect_h)

# --- Pterodactyl Class (inherits from Obstacle) ---
class Pterodactyl(Obstacle):
    def __init__(self, x_pos, ground_y, image_path, speed):
        super().__init__(x_pos, speed)
        self.image_loaded = False
        self.width = 46 # Default dimensions, adjust if needed
        self.height = 40

        try:
            if image_path and os.path.exists(image_path):
                 raw_image = pygame.image.load(image_path).convert_alpha()
                 self.image = pygame.transform.smoothscale(raw_image, (self.width, self.height))
                 self.image_loaded = True
            else: self._create_fallback()
        except Exception as e: print(f"Error loading pterodactyl: {e}"); self._create_fallback()

        # Fly at different heights
        self.y = random.choice([ground_y - 40, ground_y - 75, ground_y - 100]) # Top y coordinate

    def _create_fallback(self):
        self.image = pygame.Surface((self.width, self.height)); self.image.fill(RED)
        self.image_loaded = True

    def get_rect(self):
        padding = 4
        rect_w = max(1, self.width - 2 * padding)
        rect_h = max(1, self.height - 2 * padding)
        return pygame.Rect(self.x + padding, self.y + padding, rect_w, rect_h)

# --- Dino Class ---
class Dino:
    def __init__(self, x, ground_y_coord):
        self.x = x
        self.run_img = None
        self.duck_img = None
        self.image_loaded = False
        self.dino_width_ref = 44
        self.dino_height_ref = 47
        self.duck_height_ref = 27 # Approx height of ducking sprite

        try:
            if os.path.exists(DINO_IMG_PATH):
                self.run_img = pygame.transform.smoothscale(pygame.image.load(DINO_IMG_PATH).convert_alpha(), (self.dino_width_ref, self.dino_height_ref))
            if os.path.exists(DINO_DUCK_IMG_PATH):
                self.duck_img = pygame.transform.smoothscale(pygame.image.load(DINO_DUCK_IMG_PATH).convert_alpha(), (self.dino_width_ref + 14, self.duck_height_ref)) # Duck sprite might be wider
            if self.run_img and self.duck_img:
                self.image_loaded = True
            else: self._create_fallback_dino()
        except Exception as e: print(f"Error loading dino images: {e}"); self._create_fallback_dino()

        # Normal state dimensions
        self.width = self.run_img.get_width() if self.run_img else self.dino_width_ref
        self.height = self.run_img.get_height() if self.run_img else self.dino_height_ref

        # Base Y position (top coord when standing on ground)
        self.base_y = ground_y_coord - self.height
        self.y = self.base_y
        self.vel_y = 0

        self.is_jumping = False
        self.is_ducking = False
        self.alive = True
        self.score = 0
        self.brain = NeuralNetwork()
        self.id = random.randint(1000, 9999)

    def _create_fallback_dino(self):
        self.run_img = pygame.Surface((self.dino_width_ref, self.dino_height_ref)); self.run_img.fill(BLACK)
        self.duck_img = pygame.Surface((self.dino_width_ref + 14, self.duck_height_ref)); self.duck_img.fill(GREY)
        self.image_loaded = True

    def update(self, obstacles, game_speed):
        if not self.alive: return

        # --- Player Input / AI Decision ---
        # Get inputs for AI
        inputs = self.get_brain_inputs(obstacles, game_speed)
        output = self.brain.forward(inputs)

        # Interpret output: Jump, Duck, or Nothing
        # Fine-tune these thresholds!
        JUMP_THRESHOLD = 0.75
        DUCK_THRESHOLD = 0.25

        action_jump = False
        action_duck = False
        if output > JUMP_THRESHOLD:
            action_jump = True
        elif output < DUCK_THRESHOLD:
            action_duck = True

        # --- State Update ---
        # Handle ducking state first
        if action_duck and not self.is_jumping: # Can only duck if not jumping
            self.is_ducking = True
        else:
            self.is_ducking = False # Stop ducking if AI doesn't say so OR if jumping

        # Handle jumping state
        if action_jump and not self.is_ducking: # Can only jump if not ducking
            self.jump()

        # --- Apply Physics ---
        if self.is_jumping:
            self.vel_y += GRAVITY * simulation_speed_factor
            self.y += self.vel_y * simulation_speed_factor
        else:
            # Not jumping, stay on ground or handle duck offset
            self.vel_y = 0 # Reset velocity if somehow gained while not jumping
            if self.is_ducking:
                 self.y = self.base_y + DUCK_Y_OFFSET
            else:
                 self.y = self.base_y

        # --- Ground Check (if jumping) ---
        if self.y >= self.base_y and self.is_jumping:
            self.y = self.base_y # Snap to ground
            self.vel_y = 0
            self.is_jumping = False
            # Stop ducking if landing while duck button held/AI output low
            # self.is_ducking = False # Re-evaluated next frame by AI output

        # Update effective height/width based on state
        current_img = self.duck_img if self.is_ducking else self.run_img
        self.width = current_img.get_width()
        self.height = current_img.get_height()

        # --- Update Score ---
        self.score += 1 * simulation_speed_factor # Score based on survival time adjusted by speed

    def get_brain_inputs(self, obstacles, game_speed):
        """ Calculates normalized inputs for the neural network. """
        obs1, obs2 = self.get_next_two_obstacles(obstacles)

        # Default values if no obstacles ahead
        dist1, h1, w1 = WIDTH, 0, 0 # Far away, zero size
        dist2 = WIDTH * 1.5 # Even further

        if obs1:
            dist1 = max(0, obs1.x - (self.x + self.width))
            h1 = obs1.height
            w1 = obs1.width
            if obs2:
                dist2 = max(0, obs2.x - (self.x + self.width))
            else:
                dist2 = WIDTH * 1.5 # No second obstacle visible

        # Normalization (crucial!) - adjust divisors as needed based on observed max values
        norm_dist1 = dist1 / WIDTH
        norm_h1 = h1 / 100.0 # Max expected obstacle height approx 100?
        norm_w1 = w1 / 100.0 # Max expected obstacle width approx 100?
        norm_dist2 = dist2 / (WIDTH * 1.5)
        norm_speed = game_speed / 25.0 # Max expected speed approx 25?
        norm_vy = (self.vel_y / abs(JUMP_STRENGTH * 1.5)) # Normalize velocity relative to jump power

        return [norm_dist1, norm_h1, norm_w1, norm_dist2, norm_speed, norm_vy]


    def get_next_two_obstacles(self, obstacles):
        """ Finds the nearest two obstacles in front of the dino. """
        obstacles_ahead = []
        dino_front_x = self.x + self.width
        for obs in obstacles:
            if obs.x + obs.width > dino_front_x: # Check if any part of obstacle is ahead
                dist = obs.x - dino_front_x
                obstacles_ahead.append({'dist': dist, 'obstacle': obs})

        obstacles_ahead.sort(key=lambda item: item['dist'])

        obs1 = obstacles_ahead[0]['obstacle'] if len(obstacles_ahead) > 0 else None
        obs2 = obstacles_ahead[1]['obstacle'] if len(obstacles_ahead) > 1 else None
        return obs1, obs2

    def jump(self):
        if not self.is_jumping and not self.is_ducking:
            self.vel_y = JUMP_STRENGTH
            self.is_jumping = True
            play_sound(sound_jump)

    def duck(self): # Manual ducking for testing
         if not self.is_jumping:
             self.is_ducking = True

    def stop_duck(self): # Manual ducking for testing
        self.is_ducking = False

    def draw(self, surface):
        if self.alive:
            current_img = self.duck_img if self.is_ducking else self.run_img
            draw_pos = (int(self.x), int(self.y))
            if self.image_loaded:
                surface.blit(current_img, draw_pos)
                # Draw shadow only when running/jumping (not ducking low)
                if not self.is_ducking:
                    shadow_rect = (self.x + 10, GROUND_Y - 5, self.dino_width_ref - 20, 10)
                    pygame.draw.ellipse(surface, GREY, shadow_rect)
            # Optional: Draw hitbox
            # pygame.draw.rect(surface, RED, self.get_rect(), 1)

    def get_rect(self):
        # Adjust hitbox based on state
        if self.is_ducking:
            # Tighter hitbox when ducking
            padding_x = 5
            padding_y = 3
            rect_w = max(1, self.width - 2 * padding_x)
            rect_h = max(1, self.height - 2 * padding_y)
            # Adjust y position slightly if needed for duck sprite alignment
            rect_y = self.y + padding_y
        else:
            # Normal running/jumping hitbox
            padding_x = 8
            padding_y = 5
            rect_w = max(1, self.width - 2 * padding_x)
            rect_h = max(1, self.height - 2 * padding_y)
            rect_y = self.y + padding_y

        return pygame.Rect(self.x + padding_x, rect_y, rect_w, rect_h)

# --- Genetic Algorithm Functions ---

def select_parent_tournament(dinos, tournament_size=5):
    """ Selects a parent using tournament selection. """
    if not dinos: return None
    tournament = random.sample(dinos, min(len(dinos), tournament_size))
    winner = max(tournament, key=lambda d: d.score)
    return winner

def evolve_population(old_dinos, mutation_rate=0.1, mutation_amount=0.2):
    """ Creates a new generation using elitism, tournament selection, and crossover. """
    if not old_dinos: return []

    sorted_dinos = sorted(old_dinos, key=lambda d: d.score, reverse=True)
    best_score = sorted_dinos[0].score if sorted_dinos else 0
    print(f"Gen Best Score: {int(best_score)}")

    new_dinos = []
    # 1. Elitism: Carry over the top N%
    elite_count = max(1, int(POPULATION_SIZE * 0.1)) # Keep top 10% (at least 1)
    for i in range(elite_count):
        if i < len(sorted_dinos):
            child = Dino(50, GROUND_Y) # Create new Dino instance
            child.brain = sorted_dinos[i].brain.clone() # Clone elite brain
            # No mutation for pure elites
            new_dinos.append(child)

    # 2. Crossover + Mutation: Fill the rest
    while len(new_dinos) < POPULATION_SIZE:
        # Select two distinct parents using tournament selection
        parent1 = select_parent_tournament(sorted_dinos)
        parent2 = select_parent_tournament(sorted_dinos)
        # Ensure parents are different if possible
        while parent2 == parent1 and len(sorted_dinos) > 1:
             parent2 = select_parent_tournament(sorted_dinos)

        if not parent1 or not parent2: # Handle case where selection fails
             print("Warning: Parent selection failed. Creating random offspring.")
             child = Dino(50, GROUND_Y) # Create fresh random dino
        else:
             child = Dino(50, GROUND_Y)
             # Perform crossover (creates child brain from parents)
             NeuralNetwork.crossover(parent1.brain, parent2.brain, child.brain)
             # Mutate the offspring
             child.brain.mutate(rate=mutation_rate, amount=mutation_amount)

        new_dinos.append(child)

    return new_dinos[:POPULATION_SIZE]

# --- Persistence ---
def save_best_brain(brain, filename="best_dino_brain_enhanced.pkl"):
    try:
        with open(filename, "wb") as f: pickle.dump(brain, f)
    except Exception as e: print(f"Error saving brain: {e}")

def load_best_brain(filename="best_dino_brain_enhanced.pkl"):
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                print(f"Loading best brain from {filename}")
                return pickle.load(f)
        except Exception as e: print(f"Error loading brain: {e}"); return None
    else: print("No saved brain found."); return None

# --- Plotting ---
def plot_scores(scores, filename="scores_plot_enhanced.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(scores, marker='.', linestyle='-')
    plt.title("Best Dino Score per Generation")
    plt.xlabel("Generation"); plt.ylabel("Highest Score")
    plt.grid(True); plt.tight_layout()
    try: plt.savefig(filename)
    except Exception as e: print(f"Could not save plot: {e}")
    plt.close()

# --- Main Game Function ---
def main():
    global simulation_speed_factor # Allow modifying global speed factor

    best_brain = load_best_brain()
    dinos = [Dino(50, GROUND_Y) for _ in range(POPULATION_SIZE)]

    if best_brain:
        print("Applying loaded brain to initial population (with mutation).")
        for i, d in enumerate(dinos):
            d.brain = best_brain.clone()
            if i > 0: d.brain.mutate(rate=0.15, amount=0.3) # Mutate non-elites more initially

    obstacles = []
    obstacle_timer = 0
    generation = 1
    all_time_best_score = 0
    best_scores_history = []
    paused = False

    game_speed = 6 # Initial game speed (internal parameter)
    min_obstacle_spawn_dist = 250 # Min pixels between obstacles
    pterodactyl_chance = 0.3 # % chance next obstacle is ptero

    # --- Main Loop ---
    run = True
    while run:
        # Calculate delta time and adjust clock based on sim speed
        actual_fps = clock.tick(FPS * simulation_speed_factor)
        effective_fps = actual_fps * simulation_speed_factor # For consistent game logic speed

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == KEY_PAUSE: paused = not paused; print("Paused" if paused else "Resumed")
                if event.key == KEY_SPEED_UP: simulation_speed_factor = min(max_sim_speed, simulation_speed_factor * 1.25); print(f"Sim Speed: {simulation_speed_factor:.2f}x")
                if event.key == KEY_SLOW_DOWN: simulation_speed_factor = max(min_sim_speed, simulation_speed_factor / 1.25); print(f"Sim Speed: {simulation_speed_factor:.2f}x")
                # Manual controls for first dino (testing)
                # if dinos and dinos[0].alive:
                #      if event.key == KEY_MANUAL_JUMP: dinos[0].jump()
                #      if event.key == KEY_MANUAL_DUCK: dinos[0].duck()
            # if event.type == pygame.KEYUP: # Manual controls for first dino
                 # if dinos and dinos[0].alive:
                 #     if event.key == KEY_MANUAL_DUCK: dinos[0].stop_duck()


        if paused: continue # Skip rest of loop if paused

        # --- Game State Update ---
        game_speed = min(22, 6 + generation * 0.25) # Internal speed increases
        obstacle_spawn_time = max(30, 90 - generation * 1.2) # Time between spawns decreases

        # Obstacle Spawning
        obstacle_timer += 1 * simulation_speed_factor
        can_spawn = not obstacles or (obstacles[-1].x < WIDTH - min_obstacle_spawn_dist)
        if obstacle_timer > obstacle_spawn_time and can_spawn:
            # Decide obstacle type
            if random.random() < pterodactyl_chance:
                 new_obstacle = Pterodactyl(WIDTH, GROUND_Y, PTERODACTYL_IMG_PATH, game_speed)
            else:
                 chosen_type = random.choice(CACTUS_TYPES)
                 new_obstacle = Cactus(WIDTH, GROUND_Y, chosen_type, BASE_CACTUS_IMG_PATH, game_speed)

            obstacles.append(new_obstacle)
            obstacle_timer = 0

        # Update Obstacles
        for obs in obstacles[:]:
            obs.update()
            if obs.off_screen():
                obstacles.remove(obs)
                # Potentially award points for passing?
                # if obs.passed: play_sound(sound_point) # Example

        # Update Dinos
        all_dead = True
        active_dinos = 0
        current_gen_best_score = 0
        for dino in dinos:
            if dino.alive:
                active_dinos += 1
                dino.update(obstacles, game_speed)
                current_gen_best_score = max(current_gen_best_score, dino.score)

                # Collision Check
                dino_rect = dino.get_rect()
                for obs in obstacles:
                    if dino_rect.colliderect(obs.get_rect()):
                        dino.alive = False
                        play_sound(sound_die)
                        break # Dino dead

            if dino.alive: all_dead = False

        # --- Drawing ---
        screen.fill(SKY_BLUE) # Background
        pygame.draw.line(screen, BLACK, (0, GROUND_Y), (WIDTH, GROUND_Y), 2) # Ground

        for obs in obstacles: obs.draw(screen)
        for dino in dinos: dino.draw(screen) # Handles alive check internally

        # UI Text
        gen_text = FONT_MEDIUM.render(f"Gen: {generation}", True, BLACK)
        alive_text = FONT_MEDIUM.render(f"Alive: {active_dinos}/{POPULATION_SIZE}", True, BLACK)
        score_text = FONT_MEDIUM.render(f"Score: {int(current_gen_best_score)}", True, BLACK)
        speed_text = FONT_MEDIUM.render(f"Speed: {game_speed:.1f}", True, BLACK)
        best_overall_text = FONT_MEDIUM.render(f"Best: {int(all_time_best_score)}", True, BLACK)
        sim_speed_text = FONT_SMALL.render(f"Sim: {simulation_speed_factor:.2f}x", True, GREY)

        screen.blit(gen_text, (10, 10)); screen.blit(alive_text, (10, 40))
        screen.blit(score_text, (WIDTH - 220, 10)); screen.blit(speed_text, (WIDTH - 220, 40))
        screen.blit(best_overall_text, (WIDTH - 220, 70)); screen.blit(sim_speed_text, (WIDTH - 90, HEIGHT - 30))

        if paused:
             pause_text = FONT_MEDIUM.render("PAUSED", True, RED)
             screen.blit(pause_text, (WIDTH // 2 - pause_text.get_width() // 2, HEIGHT // 2 - 20))

        pygame.display.flip()

        # --- Check for Generation End ---
        if all_dead:
            print(f"\n--- Generation {generation} Finished ---")
            gen_best_dino = max(dinos, key=lambda d: d.score) if any(d.score > 0 for d in dinos) else None
            gen_score = gen_best_dino.score if gen_best_dino else 0

            best_scores_history.append(gen_score)
            all_time_best_score = max(all_time_best_score, gen_score)
            print(f"All Time Best: {int(all_time_best_score)}")

            if gen_best_dino and gen_score >= all_time_best_score: # Save if new best
                 print("Saving new best brain...")
                 save_best_brain(gen_best_dino.brain)

            if generation % 5 == 0: plot_scores(best_scores_history) # Plot every 5 gens

            # --- Evolve ---
            dinos = evolve_population(dinos)
            if not dinos: print("Evolution failed!"); run = False; break

            # --- Reset ---
            obstacles.clear(); obstacle_timer = 0
            generation += 1
            print(f"--- Starting Generation {generation} ---")
            # time.sleep(0.2) # Optional short pause

    # --- End of Game Loop ---
    pygame.quit()
    print("Game Over.")
    if best_scores_history:
        plot_scores(best_scores_history) # Ensure final plot is saved
        plt.show() # Show final plot window
    sys.exit()

# --- Asset Check and Run ---
if __name__ == "__main__":
    if not os.path.isdir(ASSET_FOLDER):
         print(f"ERROR: Asset folder '{ASSET_FOLDER}' not found.")
         sys.exit(1)
    # Check essential images
    essential_images = [DINO_IMG_PATH, DINO_DUCK_IMG_PATH, BASE_CACTUS_IMG_PATH, PTERODACTYL_IMG_PATH]
    missing_essential = False
    for img_path in essential_images:
         if not os.path.exists(img_path):
              print(f"ERROR: Essential image not found: {img_path}")
              missing_essential = True
    if missing_essential: sys.exit(1)
    # Sound check is done during loading

    main()