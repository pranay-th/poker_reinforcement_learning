import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


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


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self, action_space):
        return len(self.children) == action_space

    def best_child(self, exploration_weight=1.0):
        # Use UCT (Upper Confidence Bound for Trees) to select the best child
        return max(
            self.children.values(),
            key=lambda child: child.value / (child.visits + 1e-6) +
                              exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6))
        )


class MCTSAgent:
    def __init__(self, action_space, simulations=100, exploration_weight=1.0):
        self.action_space = action_space
        self.simulations = simulations
        self.exploration_weight = exploration_weight

    def search(self, env, root):
        for _ in range(self.simulations):
            node = root
            state = root.state

            # Selection
            while node.is_fully_expanded(self.action_space) and node.children:
                node = node.best_child(self.exploration_weight)
                state, _, _ = env.step(random.choice(range(self.action_space)))

            # Expansion
            if not node.is_fully_expanded(self.action_space):
                action = random.choice([a for a in range(self.action_space) if a not in node.children])
                next_state, reward, done = env.step(action)
                child_node = MCTSNode(next_state, parent=node)
                node.children[action] = child_node
                node = child_node

            # Simulation
            total_reward = self.simulate(env, node.state)

            # Backpropagation
            while node:
                node.visits += 1
                node.value += total_reward
                node = node.parent

    def simulate(self, env, state):
        # Perform a random rollout to estimate the value of the state
        total_reward = 0
        done = False
        for _ in range(10):  # Simulate up to 10 steps
            action = random.choice(range(self.action_space))
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward

    def choose_action(self, env, state):
        root = MCTSNode(state)
        self.search(env, root)
        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_action


def evaluate_mcts(episodes=1000, simulations=100, exploration_weight=1.0):
    env = PokerEnvironment()
    agent = MCTSAgent(action_space=env.action_space, simulations=simulations, exploration_weight=exploration_weight)
    rewards = []

    for episode in tqdm(range(episodes), desc="Evaluating MCTS"):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(env, state)
            state, reward, done = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    return rewards


def visualize_metrics(rewards, title="MCTS Performance Metrics"):
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
    # Evaluate MCTS
    rewards = evaluate_mcts(episodes=1000, simulations=100, exploration_weight=1.0)

    # Visualize the performance metrics
    visualize_metrics(rewards)