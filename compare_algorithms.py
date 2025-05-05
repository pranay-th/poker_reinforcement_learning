import numpy as np
import matplotlib.pyplot as plt
# from neural_networks import PokerEnvironment, ReinforcementLearningAgent, train_agent
from expected_sarsa import ExpectedSARSAAgent, PokerEnvironment
from sarsa import SARSA_PokerAgent
from Qlearning import ACTIONS, NUM_STATES, Q_table, choose_action, update_q_table, get_reward
from monte_carlo import MCTSAgent
from tqdm import tqdm  # Import tqdm for progress tracking


def evaluate_monte_carlo(episodes=1000):
    env = PokerEnvironment()
    agent = MCTSAgent(action_space=env.action_space, simulations=100)
    rewards = []

    for episode in tqdm(range(episodes), desc="Training Monte Carlo"):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(env, state)
            state, reward, done = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    return rewards


def evaluate_expected_sarsa(episodes=1000):
    env = PokerEnvironment()
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


def evaluate_sarsa(episodes=1000):
    actions = ['fold', 'check', 'call', 'raise_small', 'raise_large']
    agent = SARSA_PokerAgent(actions=actions, alpha=0.05, gamma=0.95, epsilon=0.5)
    rewards = []

    for episode in tqdm(range(episodes), desc="Training SARSA"):
        state = {
            'hand_strength': np.random.random(),
            'pot_odds': np.random.random(),
            'position': np.random.choice(['early', 'middle', 'late']),
            'opponent_aggression': np.random.random()
        }
        total_reward = 0
        done = False

        while not done:
            action_idx, action = agent.choose_action(state)
            reward = get_reward(state, action)
            next_state = {
                'hand_strength': max(0, min(1, state['hand_strength'] + np.random.uniform(-0.2, 0.2))),
                'pot_odds': max(0, min(1, state['pot_odds'] + np.random.uniform(-0.1, 0.1))),
                'position': state['position'],
                'opponent_aggression': max(0, min(1, state['opponent_aggression'] + np.random.uniform(-0.1, 0.1)))
            }
            next_action_idx, next_action = agent.choose_action(next_state)
            agent.learn(state, action_idx, reward, next_state, next_action_idx, done)
            state = next_state
            total_reward += reward
            done = np.random.random() < 0.1  # Simulated termination condition

        rewards.append(total_reward)

    return rewards


def evaluate_q_learning(episodes=1000):
    rewards = []

    for episode in tqdm(range(episodes), desc="Training Q-Learning"):
        state = np.random.randint(0, NUM_STATES)
        total_reward = 0

        for _ in range(10):  # Limit steps per episode
            action = choose_action(state)
            reward = get_reward(state, ACTIONS[action])
            total_reward += reward
            next_state = np.random.randint(0, NUM_STATES)
            update_q_table(state, action, reward, next_state)
            state = next_state

        rewards.append(total_reward)

    return rewards


def compare_two_algorithms(rewards1, rewards2, label1, label2):
    # Plot the results for two algorithms
    plt.figure(figsize=(12, 6))
    plt.plot(np.convolve(rewards1, np.ones(100) / 100, mode='valid'), label=label1)
    plt.plot(np.convolve(rewards2, np.ones(100) / 100, mode='valid'), label=label2)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title(f"Comparison: {label1} vs {label2}")
    plt.legend()
    plt.show()


def compare_algorithms(episodes=1000):
    print("Evaluating Monte Carlo...")
    mc_rewards = evaluate_monte_carlo(episodes)

    print("Evaluating Expected SARSA...")
    esarsa_rewards = evaluate_expected_sarsa(episodes)

    print("Evaluating SARSA...")
    sarsa_rewards = evaluate_sarsa(episodes)

    print("Evaluating Q-Learning...")
    qlearning_rewards = evaluate_q_learning(episodes)

    # Compare two algorithms at a time
    compare_two_algorithms(mc_rewards, esarsa_rewards, "Monte Carlo", "Expected SARSA")
    compare_two_algorithms(sarsa_rewards, qlearning_rewards, "SARSA", "Q-Learning")


if __name__ == "__main__":
    compare_algorithms(episodes=1000)