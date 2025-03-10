# 🐍 Snake Reinforcement Learning

This project trains a Deep Q-Network (DQN) to play the classic **Snake game** using **Reinforcement Learning (RL)**. The agent learns by interacting with the environment and optimizing its movements to maximize the score.

---

## 📌 Project Structure

```
├── snake-reinforcement-learning
│   ├── models
│   │   ├── dqn_agent.py      # Deep Q-Network agent implementation
│   ├── snake_game
│   │   ├── __init__.py
│   │   ├── snake_game.py     # Game environment (Pygame-based)
│   ├── train.py              # Training script for RL agent
│   ├── evaluate.py           # Evaluate the trained model
│   ├── dqn_snake.pth         # Saved trained model (generated after training)
│   ├── requirements.txt      # Required dependencies
│   ├── .gitignore            # Files to ignore in Git
│   ├── README.md             # Project documentation (this file)
```

---

## 🚀 Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/snake-reinforcement-learning.git
cd snake-reinforcement-learning
```

### **2️⃣ Create and Activate Virtual Environment**
```bash
python -m venv venv
# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🎮 Running the Game

### **Train the RL Agent** 🏋️‍♂️
Train the agent using Deep Q-Learning:
```bash
python train.py
```
📝 This script trains the agent for multiple episodes and saves the learned model as `dqn_snake.pth`.

### **Evaluate the Trained Agent** 🤖
After training, you can test the agent’s performance:
```bash
python evaluate.py
```
This will run the trained model inside the Snake game environment and let the AI play.

---

## 🛠️ Technical Details
- **Algorithm**: Deep Q-Learning (DQN)
- **Frameworks**: PyTorch, NumPy, Pygame
- **State Representation**: Distance to walls, food direction, collision risks
- **Rewards System**:
  - `+10`: Eating food
  - `-10`: Hitting a wall or itself
  - `-0.1`: Staying idle (encourages movement)

---

## 📈 Results & Performance
After training for **1000+ episodes**, the agent **learns to survive longer and maximize food collection**. The training process gradually improves the AI’s decision-making abilities.

---

## 📜 License
This project is **open-source** and available under the MIT License.

---

## 🏆 Contributions & Improvements
Feel free to contribute! Open an issue or submit a pull request if you have ideas for improvements.

🚀 **Enjoy AI-powered Snake!** 🐍
