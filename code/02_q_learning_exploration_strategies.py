"""Comparison of epsilon-greedy, SoftMax and UCB exploration for Q-learning."""

#Exploration and exploitation strategies for Q-learning

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Defining the model learning parameters
class QLearning:
    def __init__(self, state_space_size, action_space_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.1):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.visit_counts = np.zeros((state_space_size, action_space_size))  # This line is used to track the visits for UCB

    def choose_action_epsilon(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.q_table.shape[1]))  # Exploration line
        else:
            return np.argmax(self.q_table[state])  # Exploitation line

    def choose_action_softmax(self, state):
        q_values = self.q_table[state]
        exp_q = np.exp(q_values - np.max(q_values))  # It helps to avoid overflow
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(range(self.q_table.shape[1]), p=probs)

    def choose_action_ucb(self, state, step):
        if np.any(self.visit_counts[state] == 0):
            return np.random.choice(range(self.q_table.shape[1]))  # For Exploring unvisited actions
        
        total_visits = np.sum(self.visit_counts[state])
        action_visits = self.visit_counts[state]
        q_values = self.q_table[state]

        ucb_values = q_values + np.sqrt(2 * np.log(total_visits + 1) / (action_visits + 1))
        return np.argmax(ucb_values)

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        self.visit_counts[state][action] += 1  # For updating the visit count for UCB

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

class HVACEnvironment:
    def __init__(self, room_size, insulation, external_temp, hvac_power):
        self.room_size = room_size
        self.insulation = insulation
        self.external_temp = external_temp
        self.hvac_power = hvac_power
        self.current_temp = self.external_temp
        self.temp_bins = np.linspace(15, 30, 10)  

    def discretize_state(self, temp):
        if temp < self.temp_bins[0]:
            return 0
        elif temp > self.temp_bins[-1]:
            return len(self.temp_bins) - 1
        return np.digitize(temp, self.temp_bins) - 1  

    def step(self, action):
        action = max(0, min(action, 4))  # Helps to ensure action is within range
        temp_change = (self.hvac_power * (action - 2) - self.insulation * (self.current_temp - self.external_temp))
        self.current_temp += temp_change
        
        # Updated reward function
        deviation = abs(self.current_temp - 22)
        if deviation < 0.5:
            reward = 10  
        elif deviation < 1.0:
            reward = 5   
        else:
            reward = -min(deviation, 20)  
        
        discrete_state = self.discretize_state(self.current_temp)
        return discrete_state, reward

def train_agent(agent, strategy, env, num_episodes):
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state = env.discretize_state(env.current_temp)
        total_reward = 0
        length = 0
        done = False
        while not done:
            if strategy == 'epsilon':
                action = agent.choose_action_epsilon(state)
            elif strategy == 'softmax':
                action = agent.choose_action_softmax(state)
            elif strategy == 'ucb':
                action = agent.choose_action_ucb(state, episode)
                
            next_state, reward = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            length += 1
            if abs(env.current_temp - 22) < 0.5:  
                done = True
        agent.decay_epsilon()  # Decay epsilon over time for epsilon-based strategy
        episode_rewards.append(total_reward)
        episode_lengths.append(length)

    return episode_rewards, episode_lengths

def calculate_statistics(rewards, lengths):
    return {
        "average_reward": np.mean(rewards),
        "max_reward": np.max(rewards),
        "min_reward": np.min(rewards),
        "average_length": np.mean(lengths),
        "max_length": np.max(lengths),
        "min_length": np.min(lengths)
    }

def plot_comparison(rewards_epsilon, rewards_softmax, rewards_ucb, num_episodes):
    episodes = range(num_episodes)
    plt.figure(figsize=(14, 7))
    
    plt.plot(episodes, rewards_epsilon, label='Epsilon-Greedy')
    plt.plot(episodes, rewards_softmax, label='Softmax')
    plt.plot(episodes, rewards_ucb, label='UCB')
    
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Comparison of Exploration-Exploitation Strategies')
    plt.legend()
    plt.show()

def save_q_table_image(q_table, path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(q_table, annot=True, fmt=".2f", cmap='viridis')
    plt.title('Q-Table Heatmap')
    plt.xlabel('Actions')
    plt.ylabel('States')
    plt.savefig(path)
    plt.close()

# Definining the environment and agents
env = HVACEnvironment(room_size=50, insulation=0.1, external_temp=20, hvac_power=5)

agent_epsilon = QLearning(state_space_size=10, action_space_size=5, alpha=0.05, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01)
agent_softmax = QLearning(state_space_size=10, action_space_size=5, alpha=0.05, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01)
agent_ucb = QLearning(state_space_size=10, action_space_size=5, alpha=0.05, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01)

# Code for Training agents
num_episodes = 1000000
rewards_epsilon, lengths_epsilon = train_agent(agent_epsilon, 'epsilon', env, num_episodes)
rewards_softmax, lengths_softmax = train_agent(agent_softmax, 'softmax', env, num_episodes)
rewards_ucb, lengths_ucb = train_agent(agent_ucb, 'ucb', env, num_episodes)

# Calculating statistics
stats_epsilon = calculate_statistics(rewards_epsilon, lengths_epsilon)
stats_softmax = calculate_statistics(rewards_softmax, lengths_softmax)
stats_ucb = calculate_statistics(rewards_ucb, lengths_ucb)

print("Epsilon-Greedy Statistics:", stats_epsilon)
print("Softmax Statistics:", stats_softmax)
print("UCB Statistics:", stats_ucb)

# Plotting the results
plot_comparison(rewards_epsilon, rewards_softmax, rewards_ucb, num_episodes)

# Saving the Q-tables as images
q_table_epsilon_path = OUTPUT_DIR / "q_table_epsilon.png"
q_table_softmax_path = OUTPUT_DIR / "q_table_softmax.png"
q_table_ucb_path = OUTPUT_DIR / "q_table_ucb.png"

save_q_table_image(agent_epsilon.q_table, q_table_epsilon_path)
save_q_table_image(agent_softmax.q_table, q_table_softmax_path)
save_q_table_image(agent_ucb.q_table, q_table_ucb_path)

print(f"Q-tables saved as images at {q_table_epsilon_path}, {q_table_softmax_path}, and {q_table_ucb_path}")

# Printing the final Q-tables and aggregated results
print("Training completed. Final Q-tables:")
print("Epsilon-Greedy Q-table:")
print(agent_epsilon.q_table)
print("Softmax Q-table:")
print(agent_softmax.q_table)
print("UCB Q-table:")
print(agent_ucb.q_table)

print("Aggregated Results:")
print("Epsilon-Greedy:")
print(stats_epsilon)
print("Softmax:")
print(stats_softmax)
print("UCB:")
print(stats_ucb)
