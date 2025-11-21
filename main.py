import pygame
import sys
import random
import math
import time
import numpy as np
from pygame import mixer

# Initialize pygame
pygame.init()
mixer.init()

# Game settings
WIDTH, HEIGHT = 900, 700
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 100, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
PINK = (255, 105, 180)

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        # Initialize weights with random values
        self.weights_ih = np.random.randn(self.hidden_nodes, self.input_nodes) * 0.1
        self.weights_ho = np.random.randn(self.output_nodes, self.hidden_nodes) * 0.1
        
        # Biases
        self.bias_h = np.random.randn(self.hidden_nodes, 1) * 0.1
        self.bias_o = np.random.randn(self.output_nodes, 1) * 0.1
        
        self.learning_rate = 0.1
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def dsigmoid(self, y):
        return y * (1 - y)
    
    def feedforward(self, input_array):
        try:
            # Convert input to numpy array
            inputs = np.array(input_array, dtype=np.float64).reshape(-1, 1)
            
            # Hidden layer
            hidden = np.dot(self.weights_ih, inputs)
            hidden += self.bias_h
            hidden = self.sigmoid(hidden)
            
            # Output layer
            output = np.dot(self.weights_ho, hidden)
            output += self.bias_o
            output = self.sigmoid(output)
            
            return output
        except:
            return np.array([[0.5]])
    
    def train(self, input_array, target_array):
        try:
            inputs = np.array(input_array, dtype=np.float64).reshape(-1, 1)
            targets = np.array(target_array, dtype=np.float64).reshape(-1, 1)
            
            # Feedforward
            hidden = np.dot(self.weights_ih, inputs)
            hidden += self.bias_h
            hidden = self.sigmoid(hidden)
            
            outputs = np.dot(self.weights_ho, hidden)
            outputs += self.bias_o
            outputs = self.sigmoid(outputs)
            
            # Backpropagation
            output_errors = targets - outputs
            output_gradient = self.dsigmoid(outputs)
            output_gradient *= output_errors
            output_gradient *= self.learning_rate
            
            hidden_T = hidden.T
            weights_ho_deltas = np.dot(output_gradient, hidden_T)
            
            self.weights_ho += weights_ho_deltas
            self.bias_o += output_gradient
            
            hidden_errors = np.dot(self.weights_ho.T, output_errors)
            hidden_gradient = self.dsigmoid(hidden)
            hidden_gradient *= hidden_errors
            hidden_gradient *= self.learning_rate
            
            inputs_T = inputs.T
            weights_ih_deltas = np.dot(hidden_gradient, inputs_T)
            
            self.weights_ih += weights_ih_deltas
            self.bias_h += hidden_gradient
        except:
            pass

class AdvancedAIPlayer:
    def __init__(self, settings):
        self.settings = settings
        # Neural network: 5 inputs, 8 hidden, 1 output
        self.brain = NeuralNetwork(5, 8, 1)
        self.flap_cooldown = 0
        self.training_data = []
        self.generation = 1
        self.fitness = 0
        self.last_score = 0
        
    def get_inputs(self, bird, next_pipe):
        """Get normalized inputs for neural network"""
        if not next_pipe:
            return [0.5, 0.5, 0.5, 0.5, 0.5]
        
        gap_center = next_pipe['gap_y'] + next_pipe['gap_height'] / 2
        
        # Normalized inputs between 0 and 1
        inputs = [
            bird['y'] / self.settings['HEIGHT'],  # Bird Y position
            (bird['velocity'] + 15) / 30,  # Bird velocity (-15 to 15 normalized)
            max(0, (next_pipe['x'] - bird['x'])) / self.settings['WIDTH'],  # Horizontal distance
            abs(bird['y'] - gap_center) / self.settings['HEIGHT'],  # Vertical alignment
            (gap_center / self.settings['HEIGHT'])  # Gap position
        ]
        
        return inputs
    
    def should_flap(self, bird, next_pipe):
        if self.flap_cooldown > 0:
            self.flap_cooldown -= 1
            return False
            
        inputs = self.get_inputs(bird, next_pipe)
        output = self.brain.feedforward(inputs)
        
        should_flap = output[0][0] > 0.5
        
        # Store decision for training
        self.training_data.append({
            'inputs': inputs,
            'output': should_flap,
            'timestamp': time.time()
        })
        
        if should_flap:
            self.flap_cooldown = 6
            
        return should_flap
    
    def calculate_fitness(self, score, survival_time):
        """Calculate AI fitness based on performance"""
        self.fitness = score * 10 + survival_time
        return self.fitness
    
    def mutate(self, mutation_rate=0.1):
        """Mutate the neural network weights"""
        self.weights_ih += np.random.randn(*self.weights_ih.shape) * mutation_rate
        self.weights_ho += np.random.randn(*self.weights_ho.shape) * mutation_rate
        self.bias_h += np.random.randn(*self.bias_h.shape) * mutation_rate
        self.bias_o += np.random.randn(*self.bias_o.shape) * mutation_rate

class Particle:
    def __init__(self, x, y, vx, vy, lifetime, color, size):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.color = color
        self.size = size
        self.gravity = 0.1
        self.original_color = color
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.lifetime -= 1
        
        # Fade out effect
        alpha = int(255 * (self.lifetime / self.max_lifetime))
        if len(self.original_color) == 3:
            self.current_color = (*self.original_color, alpha)
        else:
            self.current_color = self.original_color
        
        return self.lifetime > 0
    
    def draw(self, surface, offset_x=0, offset_y=0):
        if hasattr(self, 'current_color'):
            pygame.draw.circle(surface, self.current_color, 
                             (int(self.x + offset_x), int(self.y + offset_y)), 
                             max(1, int(self.size * (self.lifetime / self.max_lifetime))))

class ParticleSystem:
    def __init__(self):
        self.particles = []
    
    def add_particle(self, x, y, vx, vy, lifetime, color, size):
        self.particles.append(Particle(x, y, vx, vy, lifetime, color, size))
    
    def add_explosion(self, x, y, color, count=20, speed=5):
        for _ in range(count):
            angle = random.uniform(0, math.pi * 2)
            speed_var = random.uniform(0.5, 1.5) * speed
            vx = math.cos(angle) * speed_var
            vy = math.sin(angle) * speed_var
            lifetime = random.randint(20, 60)
            size = random.uniform(2, 6)
            self.add_particle(x, y, vx, vy, lifetime, color, size)
    
    def add_trail(self, x, y, color, count=3, direction=-1):
        for _ in range(count):
            vx = random.uniform(-2, -0.5) * direction
            vy = random.uniform(-0.5, 0.5)
            lifetime = random.randint(10, 30)
            size = random.uniform(1, 3)
            self.add_particle(x, y, vx, vy, lifetime, color, size)
    
    def add_firework(self, x, y, color):
        for _ in range(50):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(2, 8)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.randint(30, 90)
            size = random.uniform(1, 4)
            self.add_particle(x, y, vx, vy, lifetime, color, size)
    
    def update(self):
        self.particles = [p for p in self.particles if p.update()]
    
    def draw(self, surface, offset_x=0, offset_y=0):
        for particle in self.particles:
            particle.draw(surface, offset_x, offset_y)

class UltraFlappyBird:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("🚀 ULTRA ADVANCED FLAPPY BIRD - AI POWERED")
        self.clock = pygame.time.Clock()
        
        # Game state
        self.game_state = "MENU"
        self.score = 0
        self.high_score = 0
        self.difficulty = "NORMAL"
        self.game_time = 0
        
        # Bird properties
        self.bird = {
            'x': 150,
            'y': HEIGHT // 2,
            'velocity': 0,
            'radius': 16,
            'flap_power': -10,
            'gravity': 0.7,
            'angle': 0,
            'flapping': False,
            'invincible': False,
            'color': YELLOW,
            'trail_timer': 0
        }
        
        # Game objects
        self.pipes = []
        self.coins = []
        self.powerups = []
        self.obstacles = []
        
        # Timers
        self.pipe_timer = 0
        self.coin_timer = 0
        self.powerup_timer = 0
        self.obstacle_timer = 0
        
        # Effects
        self.particles = ParticleSystem()
        self.screen_shake = 0
        self.flash_effect = 0
        self.zoom_level = 1.0
        self.background_offset = 0
        
        # Power-ups
        self.slow_motion = False
        self.magnet = False
        self.double_points = False
        self.shield_time = 0
        self.slow_motion_time = 0
        self.magnet_time = 0
        self.double_points_time = 0
        
        # AI System
        self.ai_player = AdvancedAIPlayer({'WIDTH': WIDTH, 'HEIGHT': HEIGHT})
        self.ai_mode = False
        self.ai_start_time = 0
        self.ai_generations = 1
        
        # Visual effects
        self.background_stars = self.generate_stars(100)
        self.parallax_layers = self.create_parallax_layers()
        
        # Game progression
        self.level = 1
        self.experience = 0
        self.skins_unlocked = ['default', 'golden', 'robot', 'phantom']
        self.current_skin = 'default'
        
        # Sound system (placeholder)
        self.sounds = {}
        self.music_playing = False
        
        # Initialize fonts
        self.fonts = {
            'title': pygame.font.Font(None, 96),
            'large': pygame.font.Font(None, 64),
            'medium': pygame.font.Font(None, 48),
            'small': pygame.font.Font(None, 32),
            'tiny': pygame.font.Font(None, 24)
        }
        
        self.start_music()
    
    def generate_stars(self, count):
        stars = []
        for _ in range(count):
            stars.append({
                'x': random.randint(0, WIDTH),
                'y': random.randint(0, HEIGHT),
                'size': random.uniform(0.5, 3),
                'brightness': random.uniform(0.3, 1.0),
                'speed': random.uniform(0.1, 0.5)
            })
        return stars
    
    def create_parallax_layers(self):
        layers = []
        colors = [(20, 20, 60), (30, 30, 80), (40, 40, 100), (50, 50, 120)]
        for i in range(4):
            layers.append({
                'speed': 0.2 + i * 0.3,
                'color': colors[i],
                'offset': 0
            })
        return layers
    
    def start_music(self):
        try:
            # Placeholder for music - in real game, load actual music files
            self.music_playing = True
        except:
            self.music_playing = False
    
    def reset_game(self):
        self.bird = {
            'x': 150,
            'y': HEIGHT // 2,
            'velocity': 0,
            'radius': 16,
            'flap_power': -10,
            'gravity': 0.7,
            'angle': 0,
            'flapping': False,
            'invincible': False,
            'color': YELLOW,
            'trail_timer': 0
        }
        
        self.pipes = []
        self.coins = []
        self.powerups = []
        self.obstacles = []
        self.score = 0
        self.game_time = 0
        self.pipe_timer = 0
        self.coin_timer = 0
        self.powerup_timer = 0
        self.obstacle_timer = 0
        
        # Reset power-ups
        self.slow_motion = False
        self.magnet = False
        self.double_points = False
        self.shield_time = 0
        self.slow_motion_time = 0
        self.magnet_time = 0
        self.double_points_time = 0
        
        if self.ai_mode:
            self.ai_start_time = time.time()
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if self.game_state == "MENU":
                    if event.key == pygame.K_SPACE:
                        self.game_state = "PLAYING"
                        self.reset_game()
                        self.ai_mode = False
                    elif event.key == pygame.K_a:
                        self.game_state = "PLAYING"
                        self.reset_game()
                        self.ai_mode = True
                        self.ai_start_time = time.time()
                    elif event.key == pygame.K_d:
                        self.difficulty = "HARD" if self.difficulty == "NORMAL" else "NORMAL"
                    elif event.key == pygame.K_s:
                        # Cycle through skins
                        current_index = self.skins_unlocked.index(self.current_skin)
                        self.current_skin = self.skins_unlocked[(current_index + 1) % len(self.skins_unlocked)]
                
                elif self.game_state == "PLAYING":
                    if event.key == pygame.K_SPACE and not self.ai_mode:
                        self.bird_flap()
                    elif event.key == pygame.K_p:
                        self.game_state = "PAUSED"
                    elif event.key == pygame.K_m:
                        self.magnet = not self.magnet
                
                elif self.game_state == "GAME_OVER":
                    if event.key == pygame.K_r:
                        self.game_state = "MENU"
                    elif event.key == pygame.K_SPACE:
                        self.game_state = "PLAYING"
                        self.reset_game()
                
                elif self.game_state == "PAUSED":
                    if event.key == pygame.K_p:
                        self.game_state = "PLAYING"
            
            if event.type == pygame.MOUSEBUTTONDOWN and self.game_state == "PLAYING" and not self.ai_mode:
                self.bird_flap()
            
            # Handle power-up timer events
            if event.type == pygame.USEREVENT + 1:
                self.bird['invincible'] = False
                self.shield_time = 0
            
            if event.type == pygame.USEREVENT + 2:
                self.slow_motion = False
                self.slow_motion_time = 0
            
            if event.type == pygame.USEREVENT + 3:
                self.magnet = False
                self.magnet_time = 0
            
            if event.type == pygame.USEREVENT + 4:
                self.double_points = False
                self.double_points_time = 0
    
    def bird_flap(self):
        self.bird['velocity'] = self.bird['flap_power']
        self.bird['flapping'] = True
        
        # Create flap particles based on skin
        particle_color = self.get_skin_particle_color()
        for _ in range(12):
            self.particles.add_particle(
                self.bird['x'] - 15, self.bird['y'],
                random.uniform(-3, -1), random.uniform(-2, 2),
                random.randint(15, 25), particle_color, random.uniform(2, 4)
            )
        
        # Screen shake for powerful flapping
        self.screen_shake = min(self.screen_shake + 2, 10)
    
    def get_skin_particle_color(self):
        skin_colors = {
            'default': (255, 255, 200),
            'golden': (255, 215, 0),
            'robot': (200, 200, 255),
            'phantom': (150, 50, 255)
        }
        return skin_colors.get(self.current_skin, (255, 255, 200))
    
    def get_bird_color(self):
        skin_colors = {
            'default': YELLOW,
            'golden': (255, 215, 0),
            'robot': (100, 100, 200),
            'phantom': (150, 50, 255)
        }
        return skin_colors.get(self.current_skin, YELLOW)
    
    def update_bird(self):
        # Apply gravity with slow motion effect
        gravity = self.bird['gravity'] * (0.5 if self.slow_motion else 1.0)
        self.bird['velocity'] += gravity
        self.bird['y'] += self.bird['velocity']
        
        # Update bird angle based on velocity
        target_angle = -self.bird['velocity'] * 3
        self.bird['angle'] += (target_angle - self.bird['angle']) * 0.15
        
        self.bird['flapping'] = False
        
        # Add trail particles
        self.bird['trail_timer'] += 1
        if self.bird['trail_timer'] >= 3:
            self.bird['trail_timer'] = 0
            trail_color = self.get_skin_particle_color()
            self.particles.add_trail(self.bird['x'], self.bird['y'], trail_color)
        
        # Screen boundaries with bounce effect
        if self.bird['y'] < 0:
            self.bird['y'] = 0
            self.bird['velocity'] = abs(self.bird['velocity']) * 0.5
            self.screen_shake = 5
        
        if self.bird['y'] > HEIGHT - 50:
            self.game_over()
    
    def generate_pipes(self):
        self.pipe_timer += 1
        pipe_interval = 80 if self.difficulty == "NORMAL" else 60
        
        if self.pipe_timer >= pipe_interval:
            self.pipe_timer = 0
            
            gap_height = 170 - min(self.level * 5, 50)  # Gets harder each level
            min_gap_y = 80 + self.level * 3
            max_gap_y = HEIGHT - gap_height - 80 - self.level * 3
            gap_y = random.randint(int(min_gap_y), int(max_gap_y))
            
            # Special pipe types based on level
            pipe_type = "normal"
            if self.level >= 3 and random.random() < 0.3:
                pipe_type = "moving"
            elif self.level >= 5 and random.random() < 0.2:
                pipe_type = "transparent"
            
            pipe = {
                'x': WIDTH,
                'gap_y': gap_y,
                'gap_height': gap_height,
                'width': 80,
                'passed': False,
                'color': self.generate_pipe_color(),
                'type': pipe_type,
                'move_direction': random.choice([-1, 1]) if pipe_type == "moving" else 0,
                'move_speed': random.uniform(0.5, 1.5),
                'alpha': 128 if pipe_type == "transparent" else 255
            }
            
            self.pipes.append(pipe)
    
    def generate_pipe_color(self):
        # Generate vibrant, changing colors based on score
        r = (100 + (self.score * 3) % 155)
        g = (150 + (self.score * 5) % 105)
        b = (200 + (self.score * 7) % 55)
        return (r, g, b)
    
    def generate_coins(self):
        self.coin_timer += 1
        if self.coin_timer >= 100 and random.random() < 0.4:
            self.coin_timer = 0
            
            coin_types = ['bronze', 'silver', 'gold']
            weights = [0.6, 0.3, 0.1]  # Probability weights
            coin_type = random.choices(coin_types, weights=weights)[0]
            
            coin = {
                'x': WIDTH,
                'y': random.randint(120, HEIGHT - 120),
                'radius': 12,
                'collected': False,
                'type': coin_type,
                'bob_offset': random.uniform(0, math.pi * 2),
                'bob_speed': random.uniform(0.05, 0.1)
            }
            
            self.coins.append(coin)
    
    def generate_powerups(self):
        self.powerup_timer += 1
        if self.powerup_timer >= 200 and random.random() < 0.25:
            self.powerup_timer = 0
            
            powerup_types = ['shield', 'slow_motion', 'magnet', 'double_points', 'nuke']
            weights = [0.25, 0.2, 0.2, 0.2, 0.15]
            powerup_type = random.choices(powerup_types, weights=weights)[0]
            
            powerup = {
                'x': WIDTH,
                'y': random.randint(120, HEIGHT - 120),
                'radius': 18,
                'type': powerup_type,
                'collected': False,
                'pulse': 0,
                'pulse_speed': random.uniform(0.1, 0.2)
            }
            
            self.powerups.append(powerup)
    
    def generate_obstacles(self):
        if self.level < 3:  # Only after level 3
            return
            
        self.obstacle_timer += 1
        if self.obstacle_timer >= 150 and random.random() < 0.2:
            self.obstacle_timer = 0
            
            obstacle_type = random.choice(['spike', 'rotator', 'bouncer'])
            
            obstacle = {
                'x': WIDTH,
                'y': random.randint(100, HEIGHT - 100),
                'radius': 20,
                'type': obstacle_type,
                'angle': 0,
                'rotation_speed': random.uniform(0.02, 0.05) if obstacle_type == 'rotator' else 0,
                'bounce_speed': random.uniform(0.5, 1.5) if obstacle_type == 'bouncer' else 0,
                'bounce_range': random.randint(50, 150) if obstacle_type == 'bouncer' else 0
            }
            
            self.obstacles.append(obstacle)
    
    def update_pipes(self):
        pipe_speed = 4 if self.difficulty == "NORMAL" else 5
        if self.slow_motion:
            pipe_speed *= 0.5
        
        for pipe in self.pipes[:]:
            pipe['x'] -= pipe_speed
            
            # Handle moving pipes
            if pipe['type'] == "moving":
                pipe['gap_y'] += pipe['move_direction'] * pipe['move_speed']
                if pipe['gap_y'] < 50 or pipe['gap_y'] > HEIGHT - pipe['gap_height'] - 50:
                    pipe['move_direction'] *= -1
            
            # Check if bird passed pipe
            if not pipe['passed'] and pipe['x'] + pipe['width'] < self.bird['x']:
                pipe['passed'] = True
                points = 2 if self.double_points else 1
                self.score += points
                self.experience += points
                
                # Level up check
                if self.experience >= self.level * 10:
                    self.level_up()
                
                # Score particles
                for _ in range(20):
                    self.particles.add_particle(
                        pipe['x'] + pipe['width'] // 2, pipe['gap_y'] + pipe['gap_height'] // 2,
                        random.uniform(-4, 4), random.uniform(-4, 4),
                        random.randint(25, 45), (255, 255, 100), random.uniform(2, 5)
                    )
            
            # Check collision
            if not self.bird['invincible']:
                if (self.bird['x'] + self.bird['radius'] > pipe['x'] and 
                    self.bird['x'] - self.bird['radius'] < pipe['x'] + pipe['width']):
                    if (self.bird['y'] - self.bird['radius'] < pipe['gap_y'] or 
                        self.bird['y'] + self.bird['radius'] > pipe['gap_y'] + pipe['gap_height']):
                        self.game_over()
                        break
            
            # Remove off-screen pipes
            if pipe['x'] + pipe['width'] < -100:
                self.pipes.remove(pipe)
    
    def update_coins(self):
        for coin in self.coins[:]:
            coin['x'] -= 4
            coin['bob_offset'] += coin['bob_speed']
            coin['y'] += math.sin(coin['bob_offset']) * 2
            
            # Apply magnet effect
            if self.magnet:
                dx = self.bird['x'] - coin['x']
                dy = self.bird['y'] - coin['y']
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < 250:  # Increased magnet radius
                    strength = min(0.3, (250 - distance) / 1000)
                    coin['x'] += dx * strength
                    coin['y'] += dy * strength
            
            # Check collection
            distance = math.sqrt((self.bird['x'] - coin['x'])**2 + (self.bird['y'] - coin['y'])**2)
            if distance < self.bird['radius'] + coin['radius'] and not coin['collected']:
                coin['collected'] = True
                
                # Score based on coin type
                base_points = {'bronze': 2, 'silver': 5, 'gold': 10}
                points = base_points[coin['type']]
                if self.double_points:
                    points *= 2
                
                self.score += points
                self.experience += points
                
                # Coin collection particles
                color_map = {
                    'bronze': (205, 127, 50),
                    'silver': (192, 192, 192),
                    'gold': (255, 215, 0)
                }
                
                self.particles.add_explosion(coin['x'], coin['y'], color_map[coin['type']], 25, 6)
                self.screen_shake = min(self.screen_shake + 3, 15)
                
                self.coins.remove(coin)
            
            # Remove off-screen coins
            elif coin['x'] + coin['radius'] < -50:
                self.coins.remove(coin)
    
    def update_powerups(self):
        for powerup in self.powerups[:]:
            powerup['x'] -= 4
            powerup['pulse'] += powerup['pulse_speed']
            
            # Check collection
            distance = math.sqrt((self.bird['x'] - powerup['x'])**2 + (self.bird['y'] - powerup['y'])**2)
            if distance < self.bird['radius'] + powerup['radius'] and not powerup['collected']:
                powerup['collected'] = True
                self.activate_powerup(powerup['type'])
                self.powerups.remove(powerup)
            
            # Remove off-screen powerups
            elif powerup['x'] + powerup['radius'] < -50:
                self.powerups.remove(powerup)
    
    def update_obstacles(self):
        for obstacle in self.obstacles[:]:
            obstacle['x'] -= 4
            
            # Update obstacle behavior
            if obstacle['type'] == 'rotator':
                obstacle['angle'] += obstacle['rotation_speed']
            elif obstacle['type'] == 'bouncer':
                obstacle['y'] += math.sin(pygame.time.get_ticks() * 0.001 * obstacle['bounce_speed']) * 2
            
            # Check collision
            if not self.bird['invincible']:
                distance = math.sqrt((self.bird['x'] - obstacle['x'])**2 + (self.bird['y'] - obstacle['y'])**2)
                if distance < self.bird['radius'] + obstacle['radius']:
                    self.game_over()
                    break
            
            # Remove off-screen obstacles
            if obstacle['x'] + obstacle['radius'] < -50:
                self.obstacles.remove(obstacle)
    
    def activate_powerup(self, powerup_type):
        self.flash_effect = 15
        self.screen_shake = 10
        
        duration = {
            'shield': 5000,
            'slow_motion': 4000,
            'magnet': 5000,
            'double_points': 6000,
            'nuke': 0
        }
        
        if powerup_type == 'shield':
            self.bird['invincible'] = True
            self.shield_time = duration['shield'] // 1000
            pygame.time.set_timer(pygame.USEREVENT + 1, duration['shield'])
            
        elif powerup_type == 'slow_motion':
            self.slow_motion = True
            self.slow_motion_time = duration['slow_motion'] // 1000
            pygame.time.set_timer(pygame.USEREVENT + 2, duration['slow_motion'])
            
        elif powerup_type == 'magnet':
            self.magnet = True
            self.magnet_time = duration['magnet'] // 1000
            pygame.time.set_timer(pygame.USEREVENT + 3, duration['magnet'])
            
        elif powerup_type == 'double_points':
            self.double_points = True
            self.double_points_time = duration['double_points'] // 1000
            pygame.time.set_timer(pygame.USEREVENT + 4, duration['double_points'])
            
        elif powerup_type == 'nuke':
            # Clear all pipes on screen
            for pipe in self.pipes[:]:
                self.particles.add_explosion(pipe['x'] + pipe['width']//2, pipe['gap_y'] + pipe['gap_height']//2, RED, 30, 8)
            self.pipes.clear()
            self.score += 10
        
        # Powerup collection particles
        color_map = {
            'shield': (0, 200, 255),
            'slow_motion': (200, 0, 255),
            'magnet': (255, 100, 0),
            'double_points': (0, 255, 100),
            'nuke': (255, 50, 50)
        }
        
        self.particles.add_explosion(self.bird['x'], self.bird['y'], color_map[powerup_type], 40, 8)
    
    def level_up(self):
        self.level += 1
        self.experience = 0
        self.flash_effect = 30
        
        # Level up effects
        self.particles.add_firework(WIDTH // 2, HEIGHT // 2, (255, 255, 0))
        self.particles.add_firework(WIDTH // 3, HEIGHT // 3, (255, 0, 255))
        self.particles.add_firework(WIDTH * 2 // 3, HEIGHT * 2 // 3, (0, 255, 255))
        
        self.screen_shake = 20
    
    def update_ai(self):
        """Update AI player decisions"""
        if not self.pipes:
            return
            
        # Find the next pipe
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + pipe['width'] > self.bird['x'] - 100:
                next_pipe = pipe
                break
        
        if next_pipe and self.ai_player.should_flap(self.bird, next_pipe):
            self.bird_flap()
    
    def game_over(self):
        self.game_state = "GAME_OVER"
        self.high_score = max(self.high_score, self.score)
        
        # AI learning
        if self.ai_mode:
            survival_time = time.time() - self.ai_start_time
            fitness = self.ai_player.calculate_fitness(self.score, survival_time)
            
            # Mutate and improve
            if fitness > self.ai_player.last_score:
                self.ai_generations += 1
                self.ai_player.generation = self.ai_generations
                self.ai_player.last_score = fitness
            else:
                self.ai_player.mutate(0.2)
        
        # Game over particles
        self.particles.add_explosion(self.bird['x'], self.bird['y'], RED, 60, 10)
        self.particles.add_explosion(self.bird['x'] - 30, self.bird['y'], ORANGE, 40, 8)
        self.particles.add_explosion(self.bird['x'] + 30, self.bird['y'], YELLOW, 40, 8)
        
        self.screen_shake = 25
    
    def update(self):
        if self.game_state == "PLAYING":
            self.game_time += 1
            
            self.update_bird()
            self.generate_pipes()
            self.generate_coins()
            self.generate_powerups()
            self.generate_obstacles()
            self.update_pipes()
            self.update_coins()
            self.update_powerups()
            self.update_obstacles()
            self.particles.update()
            
            # AI Decision Making
            if self.ai_mode:
                self.update_ai()
            
            # Update power-up timers
            if self.shield_time > 0:
                self.shield_time -= 1/60
            if self.slow_motion_time > 0:
                self.slow_motion_time -= 1/60
            if self.magnet_time > 0:
                self.magnet_time -= 1/60
            if self.double_points_time > 0:
                self.double_points_time -= 1/60
            
            # Update screen shake
            if self.screen_shake > 0:
                self.screen_shake -= 0.5
            
            # Update flash effect
            if self.flash_effect > 0:
                self.flash_effect -= 1
            
            # Update background
            self.background_offset += 0.5
            for layer in self.parallax_layers:
                layer['offset'] = (layer['offset'] - layer['speed']) % WIDTH
            
            # Update stars
            for star in self.background_stars:
                star['x'] = (star['x'] - star['speed']) % WIDTH
    
    def draw(self):
        self.screen.fill(BLACK)
        
        # Apply screen shake
        shake_x = random.randint(-int(self.screen_shake), int(self.screen_shake)) if self.screen_shake > 0 else 0
        shake_y = random.randint(-int(self.screen_shake), int(self.screen_shake)) if self.screen_shake > 0 else 0
        
        # Draw animated background
        self.draw_background(shake_x, shake_y)
        
        if self.game_state == "MENU":
            self.draw_menu()
        elif self.game_state in ["PLAYING", "PAUSED"]:
            self.draw_game(shake_x, shake_y)
        elif self.game_state == "GAME_OVER":
            self.draw_game(shake_x, shake_y)
            self.draw_game_over()
        
        # Apply flash effect
        if self.flash_effect > 0:
            flash_surf = pygame.Surface((WIDTH, HEIGHT))
            flash_surf.set_alpha(min(self.flash_effect * 15, 200))
            flash_surf.fill(WHITE)
            self.screen.blit(flash_surf, (0, 0))
        
        pygame.display.flip()
    
    def draw_background(self, shake_x, shake_y):
        # Draw stars
        for star in self.background_stars:
            brightness = int(255 * star['brightness'])
            color = (brightness, brightness, brightness)
            pos_x = star['x'] + shake_x * star['speed']
            pos_y = star['y'] + shake_y * star['speed']
            pygame.draw.circle(self.screen, color, (int(pos_x), int(pos_y)), star['size'])
        
        # Draw parallax layers
        for layer in self.parallax_layers:
            for x in range(-1, 2):
                rect_x = x * WIDTH + layer['offset'] + shake_x * layer['speed'] * 2
                pygame.draw.rect(self.screen, layer['color'], (rect_x, 0, WIDTH, HEIGHT))
    
    def draw_menu(self):
        # Animated title
        title_y = 100 + math.sin(pygame.time.get_ticks() * 0.002) * 10
        title = self.fonts['title'].render("🚀 ULTRA FLAPPY BIRD", True, YELLOW)
        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, title_y))
        
        # Subtitle
        subtitle = self.fonts['small'].render("AI POWERED EDITION", True, CYAN)
        self.screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, title_y + 80))
        
        # Instructions
        instructions = [
            "PRESS SPACE TO PLAY",
            "PRESS A FOR AI MODE", 
            f"DIFFICULTY: {self.difficulty} (PRESS D)",
            f"SKIN: {self.current_skin.upper()} (PRESS S)",
            f"HIGH SCORE: {self.high_score}",
            f"AI GENERATION: {self.ai_generations}"
        ]
        
        for i, text in enumerate(instructions):
            color = WHITE if i < 4 else (200, 200, 255)
            text_surf = self.fonts['small'].render(text, True, color)
            self.screen.blit(text_surf, (WIDTH // 2 - text_surf.get_width() // 2, 280 + i * 45))
        
        # Animated bird preview
        bird_y = 500 + math.sin(pygame.time.get_ticks() * 0.001) * 30
        self.draw_bird(WIDTH // 2, bird_y, math.sin(pygame.time.get_ticks() * 0.003) * 15, True)
        
        # Particle effects in menu
        if random.random() < 0.1:
            self.particles.add_particle(
                random.randint(0, WIDTH), random.randint(0, HEIGHT),
                random.uniform(-1, 1), random.uniform(-1, 1),
                random.randint(30, 90), random.choice([RED, GREEN, BLUE, YELLOW, CYAN]), 3
            )
        self.particles.update()
        self.particles.draw(self.screen)
    
    def draw_game(self, shake_x, shake_y):
        # Draw pipes with effects
        for pipe in self.pipes:
            alpha = pipe.get('alpha', 255)
            
            # Create pipe surface with alpha
            pipe_surface = pygame.Surface((pipe['width'], HEIGHT), pygame.SRCALPHA)
            
            # Top pipe
            pygame.draw.rect(pipe_surface, (*pipe['color'], alpha), 
                           (0, 0, pipe['width'], pipe['gap_y']))
            # Bottom pipe  
            pygame.draw.rect(pipe_surface, (*pipe['color'], alpha),
                           (0, pipe['gap_y'] + pipe['gap_height'], pipe['width'], HEIGHT))
            
            # Pipe decorations
            if pipe['type'] == "moving":
                # Moving pipe pattern
                pattern_color = (min(255, pipe['color'][0] + 50), min(255, pipe['color'][1] + 50), min(255, pipe['color'][2] + 50))
                pygame.draw.rect(pipe_surface, pattern_color, (0, pipe['gap_y'] - 10, pipe['width'], 5))
                pygame.draw.rect(pipe_surface, pattern_color, (0, pipe['gap_y'] + pipe['gap_height'] + 5, pipe['width'], 5))
            
            self.screen.blit(pipe_surface, (pipe['x'] + shake_x, 0))
        
        # Draw obstacles
        for obstacle in self.obstacles:
            color = RED if obstacle['type'] == 'spike' else ORANGE if obstacle['type'] == 'rotator' else PURPLE
            if obstacle['type'] == 'rotator':
                # Draw rotated square
                points = []
                for i in range(4):
                    angle = obstacle['angle'] + i * math.pi / 2
                    px = obstacle['x'] + math.cos(angle) * obstacle['radius']
                    py = obstacle['y'] + math.sin(angle) * obstacle['radius']
                    points.append((px + shake_x, py + shake_y))
                pygame.draw.polygon(self.screen, color, points)
            else:
                pygame.draw.circle(self.screen, color, 
                                 (int(obstacle['x'] + shake_x), int(obstacle['y'] + shake_y)), 
                                 obstacle['radius'])
        
        # Draw coins with animation
        for coin in self.coins:
            color_map = {
                'bronze': (205, 127, 50),
                'silver': (192, 192, 192), 
                'gold': (255, 215, 0)
            }
            base_color = color_map[coin['type']]
            
            # Pulsing effect
            pulse = (math.sin(coin['bob_offset']) + 1) * 0.3 + 0.7
            color = tuple(int(c * pulse) for c in base_color)
            
            pygame.draw.circle(self.screen, color, 
                             (int(coin['x'] + shake_x), int(coin['y'] + shake_y)), 
                             coin['radius'])
            pygame.draw.circle(self.screen, WHITE, 
                             (int(coin['x'] + shake_x), int(coin['y'] + shake_y)), 
                             coin['radius'] - 4, 2)
        
        # Draw power-ups with pulsing effect
        for powerup in self.powerups:
            color_map = {
                'shield': (0, 200, 255),
                'slow_motion': (200, 0, 255),
                'magnet': (255, 100, 0),
                'double_points': (0, 255, 100),
                'nuke': (255, 50, 50)
            }
            
            pulse_size = (math.sin(powerup['pulse']) + 1) * 0.2 + 0.8
            radius = int(powerup['radius'] * pulse_size)
            color = color_map[powerup['type']]
            
            pygame.draw.circle(self.screen, color, 
                             (int(powerup['x'] + shake_x), int(powerup['y'] + shake_y)), 
                             radius)
            
            # Draw power-up symbol
            symbols = {'shield': '🛡️', 'slow_motion': '⏱️', 'magnet': '🧲', 'double_points': '2X', 'nuke': '💣'}
            symbol_font = pygame.font.Font(None, 24)
            symbol = symbol_font.render(symbols[powerup['type']], True, WHITE)
            self.screen.blit(symbol, (powerup['x'] + shake_x - 12, powerup['y'] + shake_y - 12))
        
        # Draw particles
        self.particles.draw(self.screen, shake_x, shake_y)
        
        # Draw bird
        self.draw_bird(self.bird['x'] + shake_x, self.bird['y'] + shake_y, self.bird['angle'], False)
        
        # Draw HUD
        self.draw_hud()
        
        if self.game_state == "PAUSED":
            self.draw_pause_screen()
    
    def draw_bird(self, x, y, angle, is_menu=False):
        # Create bird surface
        size = 50 if is_menu else 40
        bird_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Get bird color based on skin
        bird_color = self.get_bird_color()
        
        # Bird body
        body_color = bird_color if not self.bird['invincible'] else (200, 200, 255)
        pygame.draw.circle(bird_surf, body_color, (size//2, size//2), size//2 - 5)
        
        # Bird eye
        eye_size = size // 8
        pygame.draw.circle(bird_surf, BLACK, (size//2 + size//4, size//2 - size//6), eye_size)
        pygame.draw.circle(bird_surf, WHITE, (size//2 + size//4 + 1, size//2 - size//6 - 1), eye_size//2)
        
        # Bird beak
        beak_size = size // 4
        beak_points = [
            (size//2 + beak_size, size//2),
            (size - 5, size//2 - beak_size//2), 
            (size//2 + beak_size, size//2 + beak_size//2)
        ]
        pygame.draw.polygon(bird_surf, ORANGE, beak_points)
        
        # Skin-specific features
        if self.current_skin == 'robot':
            # Robot details
            pygame.draw.rect(bird_surf, (50, 50, 50), (size//4, size//3, size//2, 2))
            pygame.draw.circle(bird_surf, (100, 100, 100), (size//2, size//2), size//6, 2)
        elif self.current_skin == 'phantom':
            # Ghostly effects
            for i in range(3):
                ghost_y = size//2 + i * 3
                pygame.draw.circle(bird_surf, (*bird_color, 150), (size//2, ghost_y), size//3 - i*2)
        
        # Rotate and draw
        rotated_bird = pygame.transform.rotate(bird_surf, angle)
        rect = rotated_bird.get_rect(center=(x, y))
        self.screen.blit(rotated_bird, rect)
        
        # Invincibility effect
        if self.bird['invincible'] and not is_menu:
            shield_radius = self.bird['radius'] + 5
            pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) * 0.3 + 0.7
            alpha = int(150 * pulse)
            shield_surf = pygame.Surface((shield_radius*2, shield_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(shield_surf, (100, 200, 255, alpha), (shield_radius, shield_radius), shield_radius, 3)
            self.screen.blit(shield_surf, (x - shield_radius, y - shield_radius))
    
    def draw_hud(self):
        # Score with shadow effect
        score_text = self.fonts['large'].render(f"{self.score}", True, WHITE)
        shadow_text = self.fonts['large'].render(f"{self.score}", True, BLACK)
        
        # Draw shadow
        self.screen.blit(shadow_text, (WIDTH//2 - score_text.get_width()//2 + 2, 32))
        self.screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, 30))
        
        # Top bar info
        top_info = [
            f"HIGH: {self.high_score}",
            f"LEVEL: {self.level}",
            f"XP: {self.experience}/{self.level * 10}"
        ]
        
        for i, text in enumerate(top_info):
            color = YELLOW if i == 0 else CYAN if i == 1 else GREEN
            text_surf = self.fonts['tiny'].render(text, True, color)
            x_pos = 20 if i == 0 else WIDTH - text_surf.get_width() - 20 if i == 2 else WIDTH//2 - text_surf.get_width()//2
            self.screen.blit(text_surf, (x_pos, 20))
        
        # Game mode indicator
        if self.ai_mode:
            mode_text = self.fonts['small'].render(f"AI MODE - GEN {self.ai_generations}", True, (0, 255, 255))
            self.screen.blit(mode_text, (WIDTH//2 - mode_text.get_width()//2, 80))
        
        # Active power-ups panel
        active_powerups = []
        if self.bird['invincible']:
            active_powerups.append(("SHIELD", f"{max(0, int(self.shield_time))}s", (0, 200, 255)))
        if self.slow_motion:
            active_powerups.append(("SLOW MO", f"{max(0, int(self.slow_motion_time))}s", (200, 0, 255)))
        if self.magnet:
            active_powerups.append(("MAGNET", f"{max(0, int(self.magnet_time))}s", (255, 100, 0)))
        if self.double_points:
            active_powerups.append(("2x POINTS", f"{max(0, int(self.double_points_time))}s", (0, 255, 100)))
        
        for i, (name, time_left, color) in enumerate(active_powerups):
            powerup_text = self.fonts['tiny'].render(f"{name} {time_left}", True, color)
            self.screen.blit(powerup_text, (20, 100 + i * 25))
    
    def draw_pause_screen(self):
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(150)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        pause_text = self.fonts['large'].render("PAUSED", True, WHITE)
        self.screen.blit(pause_text, (WIDTH//2 - pause_text.get_width()//2, HEIGHT//2 - 50))
        
        continue_text = self.fonts['small'].render("Press P to continue", True, (200, 200, 200))
        self.screen.blit(continue_text, (WIDTH//2 - continue_text.get_width()//2, HEIGHT//2 + 30))
    
    def draw_game_over(self):
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(200)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Game over title
        game_over_text = self.fonts['large'].render("GAME OVER", True, RED)
        self.screen.blit(game_over_text, (WIDTH//2 - game_over_text.get_width()//2, 150))
        
        # Score display
        score_text = self.fonts['medium'].render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, 250))
        
        # High score
        hs_text = self.fonts['medium'].render(f"High Score: {self.high_score}", True, YELLOW)
        self.screen.blit(hs_text, (WIDTH//2 - hs_text.get_width()//2, 320))
        
        # Level and XP
        level_text = self.fonts['small'].render(f"Reached Level {self.level}", True, CYAN)
        self.screen.blit(level_text, (WIDTH//2 - level_text.get_width()//2, 390))
        
        # AI stats if applicable
        if self.ai_mode:
            ai_text = self.fonts['small'].render(f"AI Generation: {self.ai_generations}", True, (0, 255, 255))
            self.screen.blit(ai_text, (WIDTH//2 - ai_text.get_width()//2, 430))
        
        # Restart instructions
        restart_text = self.fonts['small'].render("Press SPACE to restart or R for menu", True, (200, 200, 200))
        self.screen.blit(restart_text, (WIDTH//2 - restart_text.get_width()//2, 500))
    
    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

if __name__ == "__main__":
    print("🚀 Starting Ultra Advanced Flappy Bird...")
    print("🎮 Controls: SPACE to flap, P to pause, A for AI mode")
    print("⚙️  Features: Multiple skins, power-ups, AI learning, particle effects")
    game = UltraFlappyBird()
    game.run()