import pygame
import random
import numpy as np

WIDTH, HEIGHT = 400, 400
GRID_SIZE = 20

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

class SnakeGame:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.grid_size = GRID_SIZE
        self.reset()
    
    def reset(self):
        self.snake = [(WIDTH // 2, HEIGHT // 2)]
        self.food = self._place_food()
        self.direction = (0, -GRID_SIZE)
        self.score = 0
        self.done = False
        return self.get_state()

    def _place_food(self):
        while True:
            food = (random.randint(0, (WIDTH // GRID_SIZE) - 1) * GRID_SIZE, 
                    random.randint(0, (HEIGHT // GRID_SIZE) - 1) * GRID_SIZE)
            if food not in self.snake:
                return food
    
    def step(self, action):
        old_head = self.snake[0]

        if action == 0:  # Left
            self.direction = (-GRID_SIZE, 0)
        elif action == 1:  # Right
            self.direction = (GRID_SIZE, 0)
        elif action == 2:  # Up
            self.direction = (0, -GRID_SIZE)
        elif action == 3:  # Down
            self.direction = (0, GRID_SIZE)

        new_head = (old_head[0] + self.direction[0], old_head[1] + self.direction[1])

        reward = -0.1 

        if new_head == self.food:
            self.snake.insert(0, new_head)
            self.food = self._place_food()
            self.score += 1
            reward = 10

        else:
            self.snake.insert(0, new_head)
            self.snake.pop()

            if (new_head in self.snake[1:] or
                new_head[0] < 0 or new_head[0] >= WIDTH or
                new_head[1] < 0 or new_head[1] >= HEIGHT):
                reward = -10
                self.done = True

        return self.get_state(), reward, self.done

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        # Distances to walls
        left_wall = head_x / WIDTH
        right_wall = (WIDTH - head_x) / WIDTH
        top_wall = head_y / HEIGHT
        bottom_wall = (HEIGHT - head_y) / HEIGHT

        # Direction of food
        food_left = 1 if food_x < head_x else 0
        food_right = 1 if food_x > head_x else 0
        food_up = 1 if food_y < head_y else 0
        food_down = 1 if food_y > head_y else 0

        # Danger (next step collision risk)
        danger_left = (head_x - GRID_SIZE, head_y) in self.snake or head_x - GRID_SIZE < 0
        danger_right = (head_x + GRID_SIZE, head_y) in self.snake or head_x + GRID_SIZE >= WIDTH
        danger_up = (head_x, head_y - GRID_SIZE) in self.snake or head_y - GRID_SIZE < 0
        danger_down = (head_x, head_y + GRID_SIZE) in self.snake or head_y + GRID_SIZE >= HEIGHT

        state = [
            left_wall, right_wall, top_wall, bottom_wall,
            food_left, food_right, food_up, food_down,
            danger_left, danger_right, danger_up, danger_down
        ]

        return np.array(state, dtype=np.float32)

    def render(self, screen):
        pygame.event.pump()

        screen.fill(BLACK)
        for segment in self.snake:
            pygame.draw.rect(screen, GREEN, (*segment, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(screen, RED, (*self.food, GRID_SIZE, GRID_SIZE))
        
        pygame.display.flip()