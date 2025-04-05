import pygame
import sys
import os
import random
import time
import matplotlib.pyplot as plt

import config as cfg
from entities import Dino, Cactus, Pterodactyl
from evolution import evolve_population
from utils import save_best_brain, load_best_brain, plot_scores, play_sound, sound_die

pygame.init()

screen = pygame.display.set_mode((cfg.WIDTH, cfg.HEIGHT))
pygame.display.set_caption("AI Dino Plays")
clock = pygame.time.Clock()

FONT_SMALL = pygame.font.SysFont("roboto", 24)
FONT_MEDIUM = pygame.font.SysFont("roboto", 30)

simulation_speed_factor = cfg.INITIAL_SIM_SPEED

def main():
    global simulation_speed_factor

    best_brain = load_best_brain()
    dinos = [Dino(cfg.GROUND_Y) for _ in range(cfg.POPULATION_SIZE)]
    if best_brain:
        print("Applying loaded brain to initial population (with mutation).")
        for i, d in enumerate(dinos):
            d.brain = best_brain.clone()
            if i > 0: d.brain.mutate(rate=0.15, amount=0.3)

    obstacles = []
    obstacle_timer = 0
    generation = 1
    all_time_best_score = 0
    best_scores_history = []
    paused = False
    game_speed = cfg.INITIAL_GAME_SPEED

    run = True
    while run:
        actual_fps = clock.tick(cfg.FPS * simulation_speed_factor)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: run = False
            if event.type == pygame.KEYDOWN:
                if event.key == cfg.KEY_PAUSE: paused = not paused; print("Paused" if paused else "Resumed")
                if event.key == cfg.KEY_SPEED_UP: simulation_speed_factor = min(cfg.MAX_SIM_SPEED, simulation_speed_factor * cfg.SIM_SPEED_MULTIPLIER); print(f"Sim Speed: {simulation_speed_factor:.2f}x")
                if event.key == cfg.KEY_SLOW_DOWN: simulation_speed_factor = max(cfg.MIN_SIM_SPEED, simulation_speed_factor / cfg.SIM_SPEED_MULTIPLIER); print(f"Sim Speed: {simulation_speed_factor:.2f}x")

        if paused: continue

        game_speed = min(cfg.MAX_GAME_SPEED, cfg.INITIAL_GAME_SPEED + generation * cfg.GAME_SPEED_INCREMENT)
        obstacle_spawn_time = max(cfg.MIN_SPAWN_TIME, cfg.INITIAL_SPAWN_TIME - generation * cfg.SPAWN_TIME_DECREMENT)

        obstacle_timer += 1 * simulation_speed_factor
        can_spawn = not obstacles or (obstacles[-1].x < cfg.WIDTH - cfg.MIN_OBSTACLE_SPAWN_DIST)
        if obstacle_timer > obstacle_spawn_time and can_spawn:
            if random.random() < cfg.PTERODACTYL_CHANCE:
                 new_obstacle = Pterodactyl(cfg.WIDTH, cfg.GROUND_Y, cfg.PTERODACTYL_IMG_PATH, game_speed)
            else:
                 chosen_type = random.choice(cfg.CACTUS_TYPES)
                 new_obstacle = Cactus(cfg.WIDTH, cfg.GROUND_Y, chosen_type, cfg.BASE_CACTUS_IMG_PATH, game_speed)
            obstacles.append(new_obstacle)
            obstacle_timer = 0

        for obs in obstacles[:]:
            obs.update(simulation_speed_factor)
            if obs.off_screen(): obstacles.remove(obs)

        all_dead = True
        active_dinos = 0
        current_gen_best_score = 0
        for dino in dinos:
            if dino.alive:
                active_dinos += 1
                dino.update(obstacles, game_speed, simulation_speed_factor)
                current_gen_best_score = max(current_gen_best_score, dino.score)
                dino_rect = dino.get_rect()
                for obs in obstacles:
                    if dino_rect.colliderect(obs.get_rect()):
                        dino.alive = False
                        play_sound(sound_die)
                        break
            if dino.alive: all_dead = False

        screen.fill(cfg.WHITE)
        pygame.draw.line(screen, cfg.BLACK, (0, cfg.GROUND_Y), (cfg.WIDTH, cfg.GROUND_Y), 2)
        for obs in obstacles: obs.draw(screen)
        for dino in dinos: dino.draw(screen)

        gen_text = FONT_MEDIUM.render(f"Gen: {generation}", True, cfg.BLACK)
        alive_text = FONT_MEDIUM.render(f"Alive: {active_dinos}/{cfg.POPULATION_SIZE}", True, cfg.BLACK)
        score_text = FONT_MEDIUM.render(f"Score: {int(current_gen_best_score)}", True, cfg.BLACK)
        speed_text = FONT_MEDIUM.render(f"Speed: {game_speed:.1f}", True, cfg.BLACK)
        best_overall_text = FONT_MEDIUM.render(f"Best: {int(all_time_best_score)}", True, cfg.BLACK)
        sim_speed_text = FONT_SMALL.render(f"Sim: {simulation_speed_factor:.2f}x", True, cfg.GREY)
        screen.blit(gen_text, (10, 10)); screen.blit(alive_text, (10, 40))
        screen.blit(score_text, (cfg.WIDTH - 220, 10)); screen.blit(speed_text, (cfg.WIDTH - 220, 40))
        screen.blit(best_overall_text, (cfg.WIDTH - 220, 70)); screen.blit(sim_speed_text, (cfg.WIDTH - 90, cfg.HEIGHT - 30))
        if paused:
             pause_text = FONT_MEDIUM.render("PAUSED", True, cfg.RED)
             screen.blit(pause_text, (cfg.WIDTH // 2 - pause_text.get_width() // 2, cfg.HEIGHT // 2 - 20))
        pygame.display.flip()

        if all_dead:
            print(f"\n--- Generation {generation} Finished ---")
            gen_best_dino = max(dinos, key=lambda d: d.score) if any(d.score > 0 for d in dinos) else None
            gen_score = gen_best_dino.score if gen_best_dino else 0
            best_scores_history.append(gen_score)
            all_time_best_score = max(all_time_best_score, gen_score)
            print(f"All Time Best: {int(all_time_best_score)}")

            if gen_best_dino and gen_score >= all_time_best_score:
                 print("Saving new best brain...")
                 save_best_brain(gen_best_dino.brain)
            if generation % 5 == 0 or (gen_best_dino and gen_score == all_time_best_score):
                 plot_scores(best_scores_history)

            dinos = evolve_population(dinos)
            if not dinos: print("Evolution failed!"); run = False; break

            obstacles.clear(); obstacle_timer = 0
            generation += 1
            print(f"--- Starting Generation {generation} ---")

    pygame.quit()
    print("Game Over.")
    if best_scores_history:
        plot_scores(best_scores_history)
        plt.show()
    sys.exit()

if __name__ == "__main__":
    if not os.path.isdir(cfg.ASSET_FOLDER):
         print(f"ERROR: Asset folder '{cfg.ASSET_FOLDER}' not found.")
         sys.exit(1)
    essential_images = [cfg.DINO_IMG_PATH, cfg.DINO_DUCK_IMG_PATH, cfg.BASE_CACTUS_IMG_PATH, cfg.PTERODACTYL_IMG_PATH]
    missing_essential = False
    for img_path in essential_images:
         if not os.path.exists(img_path):
              print(f"ERROR: Essential image not found: {img_path}")
              missing_essential = True
    if missing_essential: sys.exit(1)
    main()