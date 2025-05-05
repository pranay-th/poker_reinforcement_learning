import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class PokerEnvironment:
    def __init__(self):
        # Define the state and action space
        self.state_space = 10  # Example state space size
        self.action_space = 3  # Example actions: fold, call, raise

    def reset(self):
        # Reset the environment to an initial state
        return np.random.rand(self.state_space)

    def step(self, action):
        # Simulate the environment's response to an action
        next_state = np.random.rand(self.state_space)
        reward = np.random.randn()  # Example reward
        done = np.random.rand() > 0.95  # Example termination condition
        return next_state, reward, done

class ReinforcementLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.001):
        self.state_space = state_space
        self.action_space = action_space
        self.model = self.build_model(learning_rate)

    def build_model(self, learning_rate):
        model = Sequential([
            Dense(24, input_dim=self.state_space, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_space, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy')
        return model

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)
        probabilities = self.model.predict(state, verbose=0)[0]
        return np.random.choice(self.action_space, p=probabilities)

    def train(self, states, actions, rewards, gamma=0.99):
        discounted_rewards = self.compute_discounted_rewards(rewards, gamma)
        actions_one_hot = tf.keras.utils.to_categorical(actions, num_classes=self.action_space)
        self.model.fit(states, actions_one_hot, sample_weight=discounted_rewards, verbose=0)

    def compute_discounted_rewards(self, rewards, gamma):
        discounted = np.zeros_like(rewards, dtype=np.float32)
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = cumulative * gamma + rewards[t]
            discounted[t] = cumulative
        return discounted

def train_agent(episodes=1000):
    env = PokerEnvironment()
    agent = ReinforcementLearningAgent(env.state_space, env.action_space)

    for episode in range(episodes):
        state = env.reset()
        states, actions, rewards = [], [], []
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        states = np.array(states)
        agent.train(states, actions, rewards)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} completed.")

def visualize_agent_play(agent, env, num_rounds=10):
    """
    Simulate and visualize the agent playing poker.
    """
    for round_num in range(num_rounds):
        state = env.reset()
        done = False
        total_reward = 0
        round_states, round_actions, round_rewards = [], [], []

        print(f"\n--- Round {round_num + 1} ---")
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            # Log the state, action, and reward
            round_states.append(state)
            round_actions.append(action)
            round_rewards.append(reward)

            # Print the action and reward
            print(f"State: {state}, Action: {action}, Reward: {reward}")

            total_reward += reward
            state = next_state

        print(f"Total Reward for Round {round_num + 1}: {total_reward}")

        # Plot the rewards for the round
        plt.plot(round_rewards, label=f'Round {round_num + 1}')

    plt.title("Rewards per Action in Each Round")
    plt.xlabel("Action Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    env = PokerEnvironment()
    agent = ReinforcementLearningAgent(env.state_space, env.action_space)

    # Train the agent
    train_agent()

    # Visualize the agent playing poker
    visualize_agent_play(agent, env)