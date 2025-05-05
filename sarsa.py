import numpy as np
import random
import matplotlib.pyplot as plt

class SARSA_PokerAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Initialize the SARSA agent for poker
        
        Parameters:
        - actions: list of possible actions (e.g., ['fold', 'call', 'raise'])
        - alpha: learning rate (0.1 by default)
        - gamma: discount factor (0.9 by default)
        - epsilon: exploration rate (0.1 by default)
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # Initialize Q-table
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'avg_rewards': [],
            'epsilon': []
        }
        
    def choose_action(self, state):
        """
        Choose an action using ε-greedy policy
        """
        state_str = str(state)  # Convert state to a string for dictionary keys
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(len(self.actions))
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.actions)), self.actions[np.random.randint(len(self.actions))]
        else:
            return np.argmax(self.q_table[state_str]), self.actions[np.argmax(self.q_table[state_str])]
    
    def learn(self, state, action_idx, reward, next_state, next_action_idx, done):
        """
        Update Q-values using SARSA algorithm
        """
        state_str = str(state)
        next_state_str = str(next_state)
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(len(self.actions))
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = np.zeros(len(self.actions))
        target = reward + (0 if done else self.gamma * self.q_table[next_state_str][next_action_idx])
        self.q_table[state_str][action_idx] += self.alpha * (target - self.q_table[state_str][action_idx])

    def record_stats(self, episode, total_reward):
        """Record training statistics for visualization"""
        self.training_stats['episodes'].append(episode)
        self.training_stats['rewards'].append(total_reward)
        
        # Calculate running average of last 100 episodes
        avg_window = 100
        if episode >= avg_window:
            avg_reward = np.mean(self.training_stats['rewards'][-avg_window:])
        else:
            avg_reward = np.mean(self.training_stats['rewards'])
        self.training_stats['avg_rewards'].append(avg_reward)
        self.training_stats['epsilon'].append(self.epsilon)

    def plot_learning_curve(self):
        """Plot the training progress"""
        plt.figure(figsize=(12, 5))
        
        # Plot rewards and average rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.training_stats['episodes'], self.training_stats['rewards'], alpha=0.3, label='Episode Reward')
        plt.plot(self.training_stats['episodes'], self.training_stats['avg_rewards'], 'r-', label='Avg Reward (100 eps)')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('SARSA Learning Progress\nPoker Strategy Optimization')
        plt.legend()
        
        # Plot epsilon decay
        plt.subplot(1, 2, 2)
        plt.plot(self.training_stats['episodes'], self.training_stats['epsilon'], 'g-')
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate Decay')
        
        plt.tight_layout()
        plt.show()

# Example usage with visualization
if __name__ == "__main__":
    # Define possible actions
    actions = ['fold', 'check', 'call', 'raise_small', 'raise_large']
    
    # Initialize SARSA agent
    agent = SARSA_PokerAgent(actions, alpha=0.05, gamma=0.95, epsilon=0.5)
    
    # Training parameters
    num_episodes = 2000
    epsilon_decay = 0.9995  # Gradually reduce exploration
    
    print("Training SARSA agent for poker strategy optimization...")
    
    for episode in range(num_episodes):
        # Reset the poker game environment and get initial state
        state = {
            'hand_strength': random.random(),
            'pot_odds': random.random(),
            'position': random.choice(['early', 'middle', 'late']),
            'opponent_aggression': random.random()
        }
        
        action_idx, action = agent.choose_action(state)
        done = False
        total_reward = 0
        
        while not done:
            # Execute action in environment (simulated here)
            if action == 'fold':
                reward = -0.5  # Small penalty for folding
            elif action == 'check':
                reward = random.uniform(-0.2, 0.3)
            elif action == 'call':
                reward = random.uniform(-0.5, 0.7) * state['hand_strength']
            elif action.startswith('raise'):
                reward = random.uniform(-0.8, 1.2) * state['hand_strength']
            
            # 10% chance to end the hand
            done = random.random() < 0.1
            
            # Get next state
            next_state = {
                'hand_strength': max(0, min(1, state['hand_strength'] + random.uniform(-0.2, 0.2))),
                'pot_odds': max(0, min(1, state['pot_odds'] + random.uniform(-0.1, 0.1))),
                'position': state['position'],  # Position changes less frequently
                'opponent_aggression': max(0, min(1, state['opponent_aggression'] + random.uniform(-0.1, 0.1)))
            }
            
            # Choose next action
            next_action_idx, next_action = agent.choose_action(next_state)
            
            # Learn from the experience
            agent.learn(state, action_idx, reward, next_state, next_action_idx, done)
            
            # Update state and action
            state = next_state
            action_idx = next_action_idx
            action = next_action
            total_reward += reward
        
        # Record statistics and decay epsilon
        agent.record_stats(episode, total_reward)
        agent.epsilon = max(0.01, agent.epsilon * epsilon_decay)
        
        # Print progress occasionally
        if episode % 200 == 0 or episode == num_episodes - 1:
            avg_reward = agent.training_stats['avg_rewards'][-1]
            print(f"Episode {episode:4d} | Total Reward: {total_reward:6.1f} | Avg Reward: {avg_reward:6.2f} | Epsilon: {agent.epsilon:.3f}")
    
    # Plot learning curves
    agent.plot_learning_curve()
    
    # Show some learned Q-values
    print("\nExample learned Q-values:")
    for state_key, q_values in list(agent.q_table.items())[:3]:
        print(f"\nState: {state_key}")
        for action, q_value in zip(actions, q_values):
            print(f"  {action:10s}: {q_value:7.3f}")