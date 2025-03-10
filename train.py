import torch
import numpy as np
from snake_game.snake_game import SnakeGame
from models.dqn_agent import DQNAgent

EPISODES = 5000

def train():
    env = SnakeGame()
    state_size = len(env.get_state())
    action_size = 4 
    agent = DQNAgent(state_size, action_size)
    
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.train()
        
        agent.update_epsilon()
        print(f"Episode {episode + 1}/{EPISODES}, Score: {total_reward}, Epsilon: {agent.epsilon:.4f}")
    
    torch.save(agent.model.state_dict(), "dqn_snake.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()