
# 🍓 Q-Learning Grid Environment using Kivy

This project demonstrates a reinforcement learning environment using the **Q-Learning** algorithm in a 2D grid world. The agent learns to navigate the grid, collect rewards (strawberries 🍓), and avoid obstacles through trial and error. The environment is visualized in real time using the **Kivy** graphical interface.

---

## 🧠 Project Overview

- **Algorithm:** Q-Learning (model-free reinforcement learning)
- **Interface:** Graphical User Interface (GUI) built with Kivy
- **Goal:** Teach an agent to find the optimal path to collect all strawberries while avoiding negative reward zones
- **Grid Size:** 10x10 grid world with:
  - ✅ Green zone (starting point)
  - 🚫 Red cells (obstacles with penalties)
  - 🍓 Strawberry cells (positive reward)
  - 🟦 Blue cell (goal/final reward — optional)

---

## 📸 Interface Screenshot

![Q-Learning Grid Screenshot](assets/image.png.png)
git push -u origin main

---

## ⚙️ Features

- Real-time training visualization
- Customizable training parameters:
  - Epsilon (exploration rate)
  - Learning rate (α)
  - Discount factor (γ)
  - Number of episodes
- Save and import Q-tables
- Track reward over time using live plotting
- Show optimal learned path

---

## 🚀 Getting Started

### 🔧 Prerequisites

Install the required dependencies:

```bash
pip install kivy matplotlib numpy
```

> ⚠️ Make sure you have Python 3.7+ installed.

---

### 🧪 Running the App

In your terminal, navigate to the project directory and run:

```bash
python main.py
```

---

## 🎛️ Interface Guide

| Component              | Description                                      |
|------------------------|--------------------------------------------------|
| Start Training         | Starts Q-learning training process               |
| Save Q-table           | Saves the learned Q-table to a file              |
| Import Q-table         | Load a previously saved Q-table                  |
| Show Optimal Path      | Displays the optimal path after training         |
| Epsilon                | Controls exploration (0 = full exploitation)     |
| Learning Rate (α)      | How quickly the agent updates its Q-values       |
| Discount Factor (γ)    | Future reward importance                         |
| Episodes               | Number of training episodes                      |

---

## 📁 Project Structure

```bash
.
├── main.py              # Main Kivy GUI and logic
├── q_learning.py        # Q-learning algorithm implementation
├── environment.py       # Environment setup (grid, rewards, penalties)
├── utils.py             # Helper functions (optional)
├── assets/              # Image files like strawberries, icons...
└── README.md            # Project description file
```

---

## 💡 Concepts Used

- Q-Learning algorithm
- ε-greedy policy for exploration/exploitation
- Value iteration and reward maximization
- Real-time rendering using Kivy framework
- State-action table (Q-table) storage and update

---

## 🛠️ To Do / Future Improvements

- Add multiple agent support
- Save training history as CSV
- Add more complex reward strategies
- Allow dynamic grid resizing

---

## 🧑‍💻 Author

 **Fatima**

Feel free to contribute, fork, or star ⭐ the repo!

---

## 📜 License

This project is licensed under the MIT License.
