"""Exploration-strategy comparison for Double Q-learning."""

#Exploration and exploitation strategies for double Q-learning

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import seaborn as sns
import os
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Defining the HVAC Environment
class HVACEnvironment:
    def __init__(self, room_size, insulation, hvac_power, external_temp):
        self.room_size = room_size
        self.insulation = insulation
        self.external_temp = external_temp
        self.hvac_power = hvac_power
        self.current_temp = self.external_temp
        self.time_of_day = 0
        self.occupancy_status = 1
        self.temp_bins = np.linspace(15, 30, 10)
        self.time_bins = np.linspace(0, 23, 24)

    def discretize_state(self):
        temp_state = np.digitize(self.current_temp, self.temp_bins) - 1
        time_state = np.digitize(self.time_of_day, self.time_bins) - 1
        return (temp_state, time_state, self.occupancy_status)

    def step(self, action):
        action = max(0, min(action, 4))
        temp_change = (self.hvac_power * (action - 2) - self.insulation * (self.current_temp - self.external_temp))
        self.current_temp += temp_change
        
        target_temp = 22 if self.occupancy_status == 1 else 18
        deviation = abs(self.current_temp - target_temp)
        
        # Adjusting the reward function to reduce the penalty for deviations beyond 1.0 degree
        if deviation < 0.5:
            reward = 10
        elif deviation < 1.0:
            reward = 5
        else:
            reward = -0.25 * (deviation - 1) 

        self.time_of_day = (self.time_of_day + 1) % 24
        self.occupancy_status = random.choice([0, 1])

        discrete_state = self.discretize_state()
        return discrete_state, reward

# Double Q-Learning Agent with Multiple Exploration Strategies
class DoubleQLearning:
    def __init__(self, state_space_dims, action_space_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, replay_size=10000, batch_size=64, strategy='epsilon'):
        self.state_space_dims = state_space_dims
        self.action_space_size = action_space_size
        self.q_table1 = np.zeros((*state_space_dims, action_space_size))
        self.q_table2 = np.zeros((*state_space_dims, action_space_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory = deque(maxlen=replay_size)
        self.batch_size = batch_size
        self.strategy = strategy
        self.visit_counts = np.zeros((*state_space_dims, action_space_size))  # Using UCB

    def choose_action(self, state):
        if self.strategy == 'epsilon':
            if random.uniform(0, 1) < self.epsilon:
                return random.choice(range(self.action_space_size))  # Exploration
            else:
                q_values = (self.q_table1[state] + self.q_table2[state]) / 2
                return np.argmax(q_values)  # Exploitation
        elif self.strategy == 'softmax':
            q_values = (self.q_table1[state] + self.q_table2[state]) / 2
            exp_q = np.exp(q_values - np.max(q_values))  # Helps to avoid overflow
            probs = exp_q / np.sum(exp_q)
            return np.random.choice(range(self.action_space_size), p=probs)
        elif self.strategy == 'ucb':
            if np.any(self.visit_counts[state] == 0):
                return np.random.choice(range(self.action_space_size))  # For exploring unvisited actions
            total_visits = np.sum(self.visit_counts[state])
            action_visits = self.visit_counts[state]
            q_values = (self.q_table1[state] + self.q_table2[state]) / 2
            ucb_values = q_values + np.sqrt(2 * np.log(total_visits + 1) / (action_visits + 1))
            return np.argmax(ucb_values)
    
    def update_q_table(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) >= self.batch_size:
            self.replay_experience()
        self.visit_counts[state][action] += 1  # For updating visit counts

#Double Q-learning works by using 2 Q-tables, it chooses actions from Q-table 1 and computes the target using Q-table 2, and the way it chooses actions from Q-table 2 and  computes the target using Q-table 1
    def replay_experience(self):
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in batch:
            # Choosing an action from Q-table 1
            best_action = np.argmax(self.q_table1[next_state])
            # Computing the target using Q-table 2
            td_target = reward + self.gamma * self.q_table2[next_state][best_action]
            td_error = td_target - self.q_table1[state][action]
            self.q_table1[state][action] += self.alpha * td_error
            
            # Choosing an action from Q-table 2
            best_action = np.argmax(self.q_table2[next_state])
            # Computing the target using Q-table 1
            td_target = reward + self.gamma * self.q_table1[next_state][best_action]
            td_error = td_target - self.q_table2[state][action]
            self.q_table2[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Training Function using multiple exploration strategies
def train_agent_double_q(strategy, external_temp, num_episodes=10000):
    state_space_dims = (10, 24, 2)
    action_space_size = 5
    agent = DoubleQLearning(state_space_dims, action_space_size, alpha=0.05, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, replay_size=10000, batch_size=64, strategy=strategy)
    env = HVACEnvironment(room_size=50, insulation=0.1, hvac_power=5, external_temp=external_temp)

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state = env.discretize_state()
        total_reward = 0
        length = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            length += 1
            if abs(env.current_temp - (22 if env.occupancy_status == 1 else 18)) < 0.5:
                done = True
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_lengths.append(length)

    return episode_rewards, episode_lengths, agent.q_table1, agent.q_table2, action_space_size

# For saving and plotting the results
def average_q_table(q_table):
    return np.mean(q_table, axis=-1)

def save_q_tables_as_csv(q_table1, q_table2, season, action_space_size):
    output_directory = OUTPUT_DIR
    os.makedirs(output_directory, exist_ok=True)

    avg_q_table1 = average_q_table(q_table1)
    avg_q_table2 = average_q_table(q_table2)

    def save_single_csv(filename, q_table):
        header = "Temp Bin, Time Bin, Average Q-Value"
        with open(filename, 'w') as file:
            file.write(header + '\n')
            for temp_bin in range(q_table.shape[0]):
                for time_bin in range(q_table.shape[1]):
                    q_value = q_table[temp_bin, time_bin]
                    if isinstance(q_value, np.ndarray):
                        q_value = q_value.flatten()[0]
                    if isinstance(q_value, (list, np.ndarray)):
                        print(f"Unexpected q_value type: {type(q_value)} with shape: {np.shape(q_value)}")
                    file.write(f"{temp_bin}, {time_bin}, {q_value:.2f}\n")

    save_single_csv(os.path.join(output_directory, f"{season}_average_q_tableQ1.csv"), avg_q_table1)
    save_single_csv(os.path.join(output_directory, f"{season}_average_q_tableQ2.csv"), avg_q_table2)

def save_q_tables_as_images(q_table1, q_table2, season, action_space_size):
    output_directory = OUTPUT_DIR
    os.makedirs(output_directory, exist_ok=True)

    avg_q_table1 = average_q_table(q_table1)
    avg_q_table2 = average_q_table(q_table2)

    for avg_q_table, table_num in [(avg_q_table1, 'Q1'), (avg_q_table2, 'Q2')]:
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_q_table[:, :, 0], cmap="coolwarm", annot=True, fmt=".2f", cbar=True)
        plt.title(f"Average Q-Table {table_num} Heatmap (Occupancy 0)")
        plt.xlabel('Time Bin')
        plt.ylabel('Temp Bin')
        plt.savefig(os.path.join(output_directory, f"{season}_average_q_table{table_num}_occupancy_0.png"))
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_q_table[:, :, 1], cmap="coolwarm", annot=True, fmt=".2f", cbar=True)
        plt.title(f"Average Q-Table {table_num} Heatmap (Occupancy 1)")
        plt.xlabel('Time Bin')
        plt.ylabel('Temp Bin')
        plt.savefig(os.path.join(output_directory, f"{season}_average_q_table{table_num}_occupancy_1.png"))
        plt.close()

# This part of the code is for training and evaluation
strategies = ['epsilon', 'softmax', 'ucb']
results = {}

for strategy in strategies:
    print(f"Training with {strategy} strategy...")
    rewards, lengths, q_table1, q_table2, action_space_size = train_agent_double_q(strategy, external_temp=np.random.uniform(-10, 30))
    results[strategy] = {
        'rewards': rewards,
        'lengths': lengths,
        'q_table1': q_table1,
        'q_table2': q_table2
    }
    save_q_tables_as_csv(q_table1, q_table2, strategy, action_space_size)
    save_q_tables_as_images(q_table1, q_table2, strategy, action_space_size)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(rewards, label=f'{strategy} Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{strategy.capitalize()}: Episode Rewards Over Time')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(lengths, label=f'{strategy} Episode Lengths', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.title(f'{strategy.capitalize()}: Episode Lengths Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Printing numerical results
    print(f"\n{strategy.capitalize()} Training Completed. Numerical Results:")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Max Reward: {np.max(rewards):.2f}")
    print(f"Min Reward: {np.min(rewards):.2f}")
    print(f"Average Episode Length: {np.mean(lengths):.2f}")
    print(f"Max Episode Length: {np.max(lengths):.2f}")
    print(f"Min Episode Length: {np.min(lengths):.2f}")
