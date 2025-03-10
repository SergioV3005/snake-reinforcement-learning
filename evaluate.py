import torch
import pygame
import numpy as np
from snake_game.snake_game import SnakeGame
from models.dqn_agent import DQN

WIDTH, HEIGHT = 400, 400

def load_model(state_size, action_size, model_path="dqn_snake.pth"):
    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    return model

def evaluate():
    pygame.init()
    pygame.display.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    env = SnakeGame()
    state_size = len(env.get_state())
    action_size = 4
    model = load_model(state_size, action_size)

    state = env.reset()
    done = False

    while not done:
        for event in pygame.event.get():  
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        env.render(screen)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_values = model(state_tensor)
            action = torch.argmax(action_values).item()

        print(f"Action taken: {action}")

        state, _, done = env.step(action)

        pygame.display.flip() 
        clock.tick(10)

    pygame.quit()
    print("Evaluation complete.")

if __name__ == "__main__":
    evaluate()