
import numpy as np
import random
import matplotlib.pyplot as plt

# Define the environment (simplified poker game)
ACTIONS = ['fold', 'call', 'raise']  # Possible actions
NUM_STATES = 10  # Simplified state space

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

def get_reward(state, action):
    """Return a reward based on state and action."""
    if action == 'fold':
        return 0  # No reward for folding
    elif action == 'call':
        return random.randint(-5, 5)  # Random moderate reward
    elif action == 'raise':
        return random.randint(-10, 10)  # Higher risk, higher reward
    return 0

# Initialize Q-table
Q_table = np.zeros((NUM_STATES, len(ACTIONS)))

def choose_action(state):
    """Choose action using epsilon-greedy policy."""
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(len(ACTIONS)))  # Explore
    else:
        return np.argmax(Q_table[state])  # Exploit

def update_q_table(state, action, reward, next_state):
    """Update Q-table using Q-learning formula."""
    best_next_action = np.max(Q_table[next_state])
    Q_table[state, action] = (1 - alpha) * Q_table[state, action] + alpha * (reward + gamma * best_next_action)

# Training loop
num_episodes = 1000
rewards_per_episode = []
for episode in range(num_episodes):
    state = random.randint(0, NUM_STATES - 1)  # Random initial state
    total_reward = 0
    for _ in range(10):  # Limit steps per episode
        action = choose_action(state)
        reward = get_reward(state, ACTIONS[action])
        total_reward += reward
        next_state = random.randint(0, NUM_STATES - 1)  # Simulated transition
        update_q_table(state, action, reward, next_state)
        state = next_state
    rewards_per_episode.append(total_reward)

# Plot rewards over episodes
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-Learning Poker Training Progress')
plt.show()

# Print trained Q-table
print("Trained Q-Table:")
print(Q_table)
