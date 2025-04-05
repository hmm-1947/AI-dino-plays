import pygame
import random
import sys
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
import os

pygame.init()

# Game settings
WIDTH, HEIGHT = 800, 400
FPS = 60
GRAVITY = 0.8
JUMP_STRENGTH = -25
POPULATION_SIZE = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Chrome Dino")
clock = pygame.time.Clock()

# Paths to images
DINO_IMG_PATH = "assets/dino.png"
CACTUS_IMG_PATH = "assets/cactus.png"

class NeuralNetwork:
    def __init__(self):
        self.input_size = 2
        self.hidden_size = 4
        self.output_size = 1
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.randn(self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.random.randn(self.output_size)

    def forward(self, inputs):
        hidden = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden = self.sigmoid(hidden)
        output = np.dot(hidden, self.weights_hidden_output) + self.bias_output
        output = self.sigmoid(output)
        return output[0]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def clone(self):
        clone = NeuralNetwork()
        clone.weights_input_hidden = np.copy(self.weights_input_hidden)
        clone.bias_hidden = np.copy(self.bias_hidden)
        clone.weights_hidden_output = np.copy(self.weights_hidden_output)
        clone.bias_output = np.copy(self.bias_output)
        return clone

    def mutate(self, rate=0.1):
        def mutate_array(arr):
            mutation_mask = np.random.rand(*arr.shape) < rate
            random_mutation = np.random.randn(*arr.shape) * 0.5
            arr += mutation_mask * random_mutation
        mutate_array(self.weights_input_hidden)
        mutate_array(self.bias_hidden)
        mutate_array(self.weights_hidden_output)
        mutate_array(self.bias_output)

class Dino:
    def __init__(self, x, initial_y, image_path=None): # initial_y is the TOP coordinate
        self.x = x
        if image_path:
             raw_image = pygame.image.load(image_path).convert_alpha()
             self.image = pygame.transform.smoothscale(raw_image, (44, 47))
        else:
             self.image = pygame.Surface((44,47)) # Placeholder if no image
             self.image.fill(BLACK)
        self.width = self.image.get_width()
        self.height = self.image.get_height()

        # base_y is the y-coordinate where the DINO'S TOP should be when it's on the ground
        self.base_y = GROUND_Y - self.height
        self.y = self.base_y # Start on the ground
        self.vel_y = 0

        self.is_jumping = False
        self.alive = True
        self.score = 0
        self.brain = NeuralNetwork()
        self.id = random.randint(1000, 9999)
        print(f"Dino {self.id} created. Top y={self.y:.1f}, Bottom y={self.y + self.height:.1f}, Base (ground) y={self.base_y:.1f}, Ground Level={GROUND_Y}")

    def update(self, cactus_list):
        if not self.alive:
            return

        self.vel_y += GRAVITY
        self.y += self.vel_y

        # Ground Check: Check if the dino's *bottom* hits the ground level
        if self.y + self.height >= GROUND_Y:
            self.y = GROUND_Y - self.height # Snap top position so bottom is on ground
            if self.vel_y > 0: # Landed
                self.vel_y = 0
                self.is_jumping = False


        # --- AI Decision ---
        next_cactus = self.get_next_cactus(cactus_list)
        # ... (rest of the AI logic as modified above) ...
        if next_cactus:
            input_distance = max(0, (next_cactus.x - (self.x + self.width))) / (WIDTH - self.x - self.width)
            input_velocity = self.vel_y / abs(JUMP_STRENGTH)
            inputs = np.array([input_distance, input_velocity]).flatten()
            output = self.brain.forward(inputs)

            # --- Debugging ---
            # Print frequently to see what the AI is thinking
            # Limit prints to avoid flooding console too much, e.g., print every 10 frames or for one dino
            # if pygame.time.get_ticks() % 10 == 0 and self.id < 1010: # Example: print for one dino often
            #     print(f"Dino {self.id} | Inputs: dist={inputs[0]:.2f}, vel={inputs[1]:.2f} | NN Output: {output:.3f} | is_jumping: {self.is_jumping} | y: {self.y:.1f}, vel_y: {self.vel_y:.1f}")
            # --- End Debugging ---

            if output > 0.5:
                # --- Debugging ---
                # Print specifically when a jump *should* be triggered
                # print(f"---------- Dino {self.id}: NN wants to JUMP! Output: {output:.3f}, is_jumping: {self.is_jumping} ----------")
                # --- End Debugging ---
                self.jump() # Attempt the jump

        # --- Update Score ---
        self.score += 1


    def get_next_cactus(self, cactus_list):
        closest_cactus = None
        min_dist = float('inf')
        for cactus in cactus_list:
            # Consider cacti that are ahead of the dino's front edge
            if cactus.x > self.x + self.width / 2: # Check based on cactus position relative to dino center/front
                 dist = cactus.x - (self.x + self.width)
                 if dist < min_dist:
                     min_dist = dist
                     closest_cactus = cactus
        return closest_cactus # Can be None if all cacti are behind


    def jump(self):
        # Check if the dino is on or very near the ground before allowing jump
        # Compare dino's *top* position (self.y) to its ground position (self.base_y)
        if not self.is_jumping and self.y >= self.base_y - 5: # Allow jump if on ground or slightly sunk in
            print(f"!!!!!!!! Dino {self.id} JUMPING! (y={self.y:.1f}, base_y={self.base_y:.1f}) Setting vel_y={JUMP_STRENGTH} !!!!!!!!")
            self.vel_y = JUMP_STRENGTH
            self.is_jumping = True

    def draw(self):
        if self.alive:
            draw_y = int(self.y) # Use int for drawing position
            if self.image:
                screen.blit(self.image, (self.x, draw_y))
                # Optional: Draw a ground line for reference
                # pygame.draw.line(screen, (255, 0, 0), (self.x, self.base_y + self.height), (self.x + self.width, self.base_y + self.height), 1)
            else:
                pygame.draw.rect(screen, BLACK, (self.x, draw_y, self.width, self.height))

    def get_rect(self):
        # Make hitbox slightly smaller/tighter to the visual sprite
        padding_x = 10
        padding_y = 5
        return pygame.Rect(self.x + padding_x, self.y + padding_y, self.width - 2*padding_x, self.height - 2*padding_y)


class Cactus:
     def __init__(self, image_path=None):
        if image_path:
            raw_image = pygame.image.load(image_path).convert_alpha()
            # Ensure cactus size is reasonable
            self.image = pygame.transform.smoothscale(raw_image, (30, 60)) # Adjusted height slightly
        else:
            self.image = None # Handle this case
        self.width = self.image.get_width() if self.image else 30
        self.height = self.image.get_height() if self.image else 60
        self.x = WIDTH
        # Make sure cactus sits ON the ground line (HEIGHT - 50)
        self.y = HEIGHT - 50 # This is the y-coordinate of the TOP of the cactus image
        self.speed = 8 # Maybe slightly slower speed initially?

     def update(self):
        self.x -= self.speed

     def draw(self):
        if self.image:
            # The y coordinate is the top of the image
            screen.blit(self.image, (self.x, self.y))
        else:
            pygame.draw.rect(screen, (0, 200, 0), (self.x, self.y, self.width, self.height))

     def off_screen(self):
        return self.x + self.width < 0

     def get_rect(self):
        # Make hitbox slightly forgiving
        padding = 3
        return pygame.Rect(self.x + padding, self.y + padding, self.width - 2*padding, self.height - 2*padding)

# --- Adjust Dino Y Initialization ---
# Make sure the initial Y position matches the ground level logic
GROUND_Y = HEIGHT - 50 # Define ground level clearly

# ... inside main() ...
# When creating dinos:
dinos = [Dino(50, GROUND_Y - 60, image_path=DINO_IMG_PATH) for _ in range(POPULATION_SIZE)] # Start dino with its feet on the ground
# The dino's y coordinate is its top-left corner. If ground is at GROUND_Y,
# and dino height is H, its top should be at GROUND_Y - H to stand on the ground.
# Let's re-check Dino init and update based on this.
class Cactus:
    def __init__(self, image_path=None):
        if image_path:
            raw_image = pygame.image.load(image_path).convert_alpha()
            self.image = pygame.transform.smoothscale(raw_image, (30, 50))
        else:
            self.image = None
        self.width = self.image.get_width() if self.image else 30
        self.height = self.image.get_height() if self.image else 50
        self.x = WIDTH
        self.y = HEIGHT - 50
        self.speed = 10

    def update(self):
        self.x -= self.speed

    def draw(self):
        if self.image:
            screen.blit(self.image, (self.x, self.y - self.height))
        else:
            pygame.draw.rect(screen, (0, 200, 0), (self.x, self.y - self.height, self.width, self.height))

    def off_screen(self):
        return self.x + self.width < 0

    def get_rect(self):
        padding = 5
        return pygame.Rect(self.x + padding, self.y - self.height + padding, self.width - 2*padding, self.height - 2*padding)

def evolve_population(old_dinos):
    sorted_dinos = sorted(old_dinos, key=lambda d: d.score, reverse=True)
    top_dinos = sorted_dinos[:POPULATION_SIZE // 2]
    print("Top score:", top_dinos[0].score)
    new_dinos = []
    for parent in top_dinos:
        for _ in range(2):
            child = Dino(50, HEIGHT - 100, image_path=DINO_IMG_PATH)
            child.brain = parent.brain.clone()
            child.brain.mutate(rate=0.1)
            new_dinos.append(child)
        if len(new_dinos) >= POPULATION_SIZE:
            break
    return new_dinos[:POPULATION_SIZE]

def save_best_brain(brain, filename="best_brain.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(brain, f)
    print(f"Saved best brain to {filename}")

def load_best_brain(filename="best_brain.pkl"):
    try:
        with open(filename, "rb") as f:
            print(f"Loaded best brain from {filename}")
            return pickle.load(f)
    except FileNotFoundError:
        print("No saved brain found. Starting fresh.")
        return None

def plot_scores(scores):
    plt.clf()
    plt.title("Top Dino Score by Generation")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.plot(scores, label="Best Score")
    plt.legend()
    plt.pause(0.01)

def main():
    best_brain = load_best_brain()
    # Correct Initialization using GROUND_Y
    dino_height_approx = 47 # Approximate height from the scaled image
    dinos = [Dino(50, GROUND_Y - dino_height_approx, image_path=DINO_IMG_PATH) for _ in range(POPULATION_SIZE)]

    if best_brain:
        print("Applying loaded best brain to all dinos.")
        for d in dinos:
            d.brain = best_brain.clone() # Give clones to all
            d.brain.mutate(rate=0.05) # Apply slight mutation even to loaded brain offspring
        dinos[0].brain = best_brain.clone() # Keep one exact copy of the best brain

    cacti = []
    cactus_timer = 0
    cactus_spawn_time = 90 # Initial spawn time
    generation = 1
    best_scores = []
    plt.ion()

    run = True
    while run:
        clock.tick(FPS)
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            # Optional: Manual jump for testing physics
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_SPACE:
            #         if dinos and dinos[0].alive: # Make the first dino jump manually
            #             print("Manual Jump Triggered")
            #             dinos[0].jump()


        # --- Game Speed Increase (Optional) ---
        current_speed = 8 + generation # Speed increases slightly each generation
        cactus_spawn_time = max(40, 90 - generation * 2) # Spawn cacti faster over time

        # --- Cactus Logic ---
        cactus_timer += 1
        if cactus_timer > cactus_spawn_time:
             new_cactus = Cactus(image_path=CACTUS_IMG_PATH)
             new_cactus.speed = current_speed # Set current speed
             cacti.append(new_cactus)
             cactus_timer = 0 # Reset timer

        # Update and draw cacti
        for cactus in cacti[:]:
            cactus.update() # Will use its internal speed
            cactus.draw()
            if cactus.off_screen():
                cacti.remove(cactus)


        # --- Dino Logic ---
        all_dead = True
        active_dinos = 0
        for dino in dinos:
            if dino.alive:
                active_dinos += 1
                dino.update(cacti) # Pass current speed maybe? No, update handles itself
                dino.draw()
                # Collision Check
                for cactus in cacti:
                    if dino.get_rect().colliderect(cactus.get_rect()):
                        # print(f"Dino {dino.id} collided!")
                        dino.alive = False
                        break # No need to check other cacti for this dino
            if dino.alive:
                all_dead = False

        # --- Drawing ---
        pygame.draw.line(screen, BLACK, (0, GROUND_Y), (WIDTH, GROUND_Y), 2) # Draw ground line
        font = pygame.font.SysFont(None, 28)
        gen_text = font.render(f"Generation: {generation}", True, BLACK)
        alive_text = font.render(f"Alive: {active_dinos}/{POPULATION_SIZE}", True, BLACK)
        screen.blit(gen_text, (10, 10))
        screen.blit(alive_text, (10, 40))

        # Display score of the first dino (or best current)
        if dinos:
            score_text = font.render(f"Score: {dinos[0].score}", True, BLACK)
            screen.blit(score_text, (WIDTH - 150, 10))


        pygame.display.flip()

        # --- Evolution ---
        if all_dead:
            print(f"Generation {generation} finished.")
            best_dino = max(dinos, key=lambda d: d.score)
            best_scores.append(best_dino.score)
            # plot_scores(best_scores) # Moved plotting out of loop maybe? Or keep it?

            # Only save if the score is meaningfully high? Or always save best of gen?
            if best_dino.score > 100: # Example threshold
                 save_best_brain(best_dino.brain)

            dinos = evolve_population(dinos)
            cacti.clear() # Clear obstacles
            cactus_timer = 0 # Reset spawn timer
            generation += 1
            print(f"--- Starting Generation {generation} ---")
            # Reset speeds? Or let them increase? (Handled by current_speed)

    pygame.quit()
    # --- Final Plot ---
    plt.ioff()
    plot_scores(best_scores) # Plot final scores
    plt.savefig("scores_plot.png") # Save the plot
    plt.show() # Display plot window
    sys.exit()

if __name__ == "__main__":
    # Make sure assets folder is accessible
    if not os.path.exists(DINO_IMG_PATH) or not os.path.exists(CACTUS_IMG_PATH):
        print("ERROR: Asset file not found!")
        print(f"Looked for {DINO_IMG_PATH} and {CACTUS_IMG_PATH}")
        sys.exit(1)
    main()