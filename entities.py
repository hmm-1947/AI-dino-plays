import pygame
import random
import os
import numpy as np
import config as cfg
from neural_network import NeuralNetwork
from utils import play_sound, sounds_loaded, sound_jump

class Obstacle:
    def __init__(self, x_pos, speed):
        self.x = x_pos
        self.speed = speed
        self.image = None
        self.width = 0
        self.height = 0
        self.passed = False

    def update(self, sim_speed):
        self.x -= self.speed * sim_speed

    def draw(self, surface):
        if self.image:
            surface.blit(self.image, (int(self.x), int(self.y)))

    def off_screen(self):
        return self.x + self.width < 0

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

class Cactus(Obstacle):
    def __init__(self, x_pos, ground_y, cactus_info, base_image_path, speed):
        super().__init__(x_pos, speed)
        self.image_loaded = False
        self.width = 30
        self.height = 60
        try:
            if 'scale' in cactus_info and base_image_path and os.path.exists(base_image_path):
                base_image = pygame.image.load(base_image_path).convert_alpha()
                self.width, self.height = cactus_info['scale']
                self.image = pygame.transform.smoothscale(base_image, (self.width, self.height))
                self.image_loaded = True
            else: self._create_fallback()
        except Exception as e: print(f"Error loading/scaling cactus: {e}"); self._create_fallback()
        self.y = ground_y - self.height

    def _create_fallback(self):
        self.width = self.width if hasattr(self, 'width') else 30
        self.height = self.height if hasattr(self, 'height') else 60
        self.image = pygame.Surface((self.width, self.height)); self.image.fill(cfg.GREEN)
        self.image_loaded = True

    def get_rect(self):
        padding = 3
        rect_w = max(1, self.width - 2 * padding)
        rect_h = max(1, self.height - 2 * padding)
        return pygame.Rect(self.x + padding, self.y + padding, rect_w, rect_h)

class Pterodactyl(Obstacle):
    def __init__(self, x_pos, ground_y, image_path, speed):
        super().__init__(x_pos, speed)
        self.image_loaded = False
        self.width = 46
        self.height = 40
        try:
            if image_path and os.path.exists(image_path):
                 raw_image = pygame.image.load(image_path).convert_alpha()
                 self.image = pygame.transform.smoothscale(raw_image, (self.width, self.height))
                 self.image_loaded = True
            else: self._create_fallback()
        except Exception as e: print(f"Error loading pterodactyl: {e}"); self._create_fallback()
        self.y = random.choice([ground_y - 40, ground_y - 75, ground_y - 100])

    def _create_fallback(self):
        self.image = pygame.Surface((self.width, self.height)); self.image.fill(cfg.RED)
        self.image_loaded = True

    def get_rect(self):
        padding = 4
        rect_w = max(1, self.width - 2 * padding)
        rect_h = max(1, self.height - 2 * padding)
        return pygame.Rect(self.x + padding, self.y + padding, rect_w, rect_h)


class Dino:
    INITIAL_X = 50

    def __init__(self, ground_y_coord):
        self.x = self.INITIAL_X
        self.run_img = None
        self.duck_img = None
        self.image_loaded = False
        self.dino_width_ref = 44
        self.dino_height_ref = 47
        self.duck_height_ref = 27
        try:
            if os.path.exists(cfg.DINO_IMG_PATH):
                self.run_img = pygame.transform.smoothscale(pygame.image.load(cfg.DINO_IMG_PATH).convert_alpha(), (self.dino_width_ref, self.dino_height_ref))
            if os.path.exists(cfg.DINO_DUCK_IMG_PATH):
                self.duck_img = pygame.transform.smoothscale(pygame.image.load(cfg.DINO_DUCK_IMG_PATH).convert_alpha(), (self.dino_width_ref + 14, self.duck_height_ref))
            if self.run_img and self.duck_img:
                self.image_loaded = True
            else: self._create_fallback_dino()
        except Exception as e: print(f"Error loading dino images: {e}"); self._create_fallback_dino()

        self.width = self.run_img.get_width() if self.run_img else self.dino_width_ref
        self.height = self.run_img.get_height() if self.run_img else self.dino_height_ref
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
        self.run_img = pygame.Surface((self.dino_width_ref, self.dino_height_ref)); self.run_img.fill(cfg.BLACK)
        self.duck_img = pygame.Surface((self.dino_width_ref + 14, self.duck_height_ref)); self.duck_img.fill(cfg.GREY)
        self.image_loaded = True

    def update(self, obstacles, game_speed, sim_speed):
        if not self.alive: return
        inputs = self.get_brain_inputs(obstacles, game_speed)
        output = self.brain.forward(inputs)

        action_jump = False
        action_duck = False
        if output > cfg.JUMP_THRESHOLD: action_jump = True
        elif output < cfg.DUCK_THRESHOLD: action_duck = True

        if action_duck and not self.is_jumping: self.is_ducking = True
        else: self.is_ducking = False

        if action_jump and not self.is_ducking: self.jump()

        if self.is_jumping:
            self.vel_y += cfg.GRAVITY * sim_speed
            self.y += self.vel_y * sim_speed
        else:
            self.vel_y = 0
            if self.is_ducking: self.y = self.base_y + cfg.DUCK_Y_OFFSET
            else: self.y = self.base_y

        if self.y >= self.base_y and self.is_jumping:
            self.y = self.base_y
            self.vel_y = 0
            self.is_jumping = False

        current_img = self.duck_img if self.is_ducking else self.run_img
        self.width = current_img.get_width()
        self.height = current_img.get_height()
        self.score += 1 * sim_speed

    def get_brain_inputs(self, obstacles, game_speed):
        obs1, obs2 = self.get_next_two_obstacles(obstacles)
        dist1, h1, w1 = cfg.WIDTH, 0, 0
        dist2 = cfg.WIDTH * 1.5
        if obs1:
            dist1 = max(0, obs1.x - (self.x + self.width))
            h1 = obs1.height
            w1 = obs1.width
            if obs2: dist2 = max(0, obs2.x - (self.x + self.width))
            else: dist2 = cfg.WIDTH * 1.5
        norm_dist1 = dist1 / cfg.WIDTH
        norm_h1 = h1 / 100.0
        norm_w1 = w1 / 100.0
        norm_dist2 = dist2 / (cfg.WIDTH * 1.5)
        norm_speed = game_speed / cfg.MAX_GAME_SPEED
        norm_vy = (self.vel_y / abs(cfg.JUMP_STRENGTH * 1.5))
        return [norm_dist1, norm_h1, norm_w1, norm_dist2, norm_speed, norm_vy]

    def get_next_two_obstacles(self, obstacles):
        obstacles_ahead = []
        dino_front_x = self.x + self.width
        for obs in obstacles:
            if obs.x + obs.width > dino_front_x:
                dist = obs.x - dino_front_x
                obstacles_ahead.append({'dist': dist, 'obstacle': obs})
        obstacles_ahead.sort(key=lambda item: item['dist'])
        obs1 = obstacles_ahead[0]['obstacle'] if len(obstacles_ahead) > 0 else None
        obs2 = obstacles_ahead[1]['obstacle'] if len(obstacles_ahead) > 1 else None
        return obs1, obs2

    def jump(self):
        if not self.is_jumping and not self.is_ducking:
            self.vel_y = cfg.JUMP_STRENGTH
            self.is_jumping = True
            play_sound(sound_jump)

    def duck(self):
         if not self.is_jumping: self.is_ducking = True

    def stop_duck(self):
        self.is_ducking = False

    def draw(self, surface):
        if self.alive:
            current_img = self.duck_img if self.is_ducking else self.run_img
            draw_pos = (int(self.x), int(self.y))
            if self.image_loaded:
                surface.blit(current_img, draw_pos)
                if not self.is_ducking:
                    shadow_rect = (self.x + 10, cfg.GROUND_Y - 5, self.dino_width_ref - 20, 10)
                    pygame.draw.ellipse(surface, cfg.GREY, shadow_rect)

    def get_rect(self):
        if self.is_ducking:
            padding_x = 5; padding_y = 3
            rect_w = max(1, self.width - 2 * padding_x)
            rect_h = max(1, self.height - 2 * padding_y)
            rect_y = self.y + padding_y
        else:
            padding_x = 8; padding_y = 5
            rect_w = max(1, self.width - 2 * padding_x)
            rect_h = max(1, self.height - 2 * padding_y)
            rect_y = self.y + padding_y
        return pygame.Rect(self.x + padding_x, rect_y, rect_w, rect_h)