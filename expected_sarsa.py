import numpy as np
import random
from tqdm import tqdm  # For progress tracking
import matplotlib.pyplot as plt


class PokerEnvironment:
    def __init__(self):
        self.state_space = 10  # Example state space size
        self.action_space = 4  # Example action space size (e.g., fold, check, call, raise)
        self.reset()

    def reset(self):
        # Reset the environment to an initial state
        self.state = np.random.randint(0, self.state_space)
        return self.state

    def step(self, action):
        # Simulate the environment's response to an action
        next_state = np.random.randint(0, self.state_space)
        reward = np.random.uniform(-1, 1)  # Example reward
        done = np.random.random() < 0.1  # Example termination condition
        return next_state, reward, done


class ExpectedSARSAAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((state_size, action_size))  # Initialize Q-table

    def policy(self, state):
        # Epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state, done):
        # Update Q-value using the Expected SARSA formula
        if done:
            target = reward
        else:
            expected_q = np.dot(self.q_table[next_state], self.get_action_probabilities(next_state))
            target = reward + self.gamma * expected_q

        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

    def get_action_probabilities(self, state):
        # Calculate action probabilities for the epsilon-greedy policy
        probabilities = np.ones(self.action_size) * (self.epsilon / self.action_size)
        best_action = np.argmax(self.q_table[state])
        probabilities[best_action] += (1 - self.epsilon)
        return probabilities


def train_expected_sarsa_agent(env, agent, num_episodes=1000):
    rewards = []
    for episode in tqdm(range(num_episodes), desc="Training Expected SARSA"):
        state = env.reset()  # Reset environment
        total_reward = 0
        done = False

        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed. Total Reward: {total_reward:.2f}")

    print("Training completed.")
    return rewards


def evaluate_expected_sarsa(episodes=1000):
    env = PokerEnvironment()  # Use PokerEnvironment directly
    agent = ExpectedSARSAAgent(state_size=env.state_space, action_size=env.action_space)
    rewards = []

    for episode in tqdm(range(episodes), desc="Training Expected SARSA"):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

        rewards.append(total_reward)

    return rewards


def visualize_training(rewards, title="Expected SARSA Training Progress"):
    # Plot the rewards over episodes
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Total Reward per Episode", alpha=0.6)
    plt.plot(np.convolve(rewards, np.ones(100) / 100, mode='valid'), label="Smoothed Reward (100 episodes)", color='red')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Create environment and agent
    env = PokerEnvironment()
    agent = ExpectedSARSAAgent(state_size=env.state_space, action_size=env.action_space)

    # Train the agent
    rewards = train_expected_sarsa_agent(env, agent, num_episodes=1000)

    # Visualize the training progress
    visualize_training(rewards)

    # Print final Q-table
    print("Final Q-Table:")
    print(agent.q_table)