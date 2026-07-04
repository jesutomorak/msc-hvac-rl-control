"""Seasonal Double Q-learning HVAC temperature-control experiment."""

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import seaborn as sns
import os
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Deffining the HVAC Environment
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
        
        # The reward function in this model was designed in a way to reduce the penalty for deviations beyond 1.0 degree as opposed to the other models that had larger penalties for deviations beyond 1.0 degree
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

class DoubleQLearning:
    def __init__(self, state_space_dims, action_space_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, replay_size=10000, batch_size=64):
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

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_space_size))  # This line of the code tells the model to explore
        else:
            q_values = (self.q_table1[state] + self.q_table2[state]) / 2
            return np.argmax(q_values)  # This line of the code tells the model to exploit when not exploring

    def update_q_table(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) >= self.batch_size:
            self.replay_experience()

#Double Q-learning works by using 2 Q-tables, it chooses actions from Q-table 1 and computes the target using Q-table 2, and the way it chooses actions from Q-table 2 and  computes the target using Q-table 1
    def replay_experience(self):
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in batch:
            # This line is designed to Choose actions from Q-table 1
            best_action = np.argmax(self.q_table1[next_state])
            # This line is designed to Compute targets using Q-table 2
            td_target = reward + self.gamma * self.q_table2[next_state][best_action]
            td_error = td_target - self.q_table1[state][action]
            self.q_table1[state][action] += self.alpha * td_error
            
            # This line is designed to Choose action from Q-table 2
            best_action = np.argmax(self.q_table2[next_state])
            # This line is designed to Compute target using Q-table 1
            td_target = reward + self.gamma * self.q_table1[next_state][best_action]
            td_error = td_target - self.q_table2[state][action]
            self.q_table2[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def train_agent_double_q(external_temp, num_episodes=10000):
    state_space_dims = (10, 24, 2)
    action_space_size = 5
    agent = DoubleQLearning(state_space_dims, action_space_size, alpha=0.05, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, replay_size=10000, batch_size=64)
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

def average_q_table(q_table):
    # The average Q-values across all actions and occupancy statuses is calculated in this part of the code
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
                    # Ensuring that the q-value is a scalar
                    q_value = q_table[temp_bin, time_bin]
                    if isinstance(q_value, np.ndarray):
                        q_value = q_value.flatten()[0] 
                    if isinstance(q_value, (list, np.ndarray)):
                        print(f"Unexpected q_value type: {type(q_value)} with shape: {np.shape(q_value)}")
                    file.write(f"{temp_bin}, {time_bin}, {q_value:.2f}\n")

    # The average Q-tables were saved as CSV files
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
        filename = os.path.join(output_directory, f"{season}_average_q_table{table_num}_occupancy_0.png")
        plt.savefig(filename)
        print(f"Saved: {filename}")
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_q_table[:, :, 1], cmap="coolwarm", annot=True, fmt=".2f", cbar=True)
        filename = os.path.join(output_directory, f"{season}_average_q_table{table_num}_occupancy_1.png")
        plt.savefig(filename)
        print(f"Saved: {filename}")
        plt.close()

# This line of code runs the training for the winter and summer with Double Q-Learning
winter_rewards, winter_lengths, winter_q_table1, winter_q_table2, action_space_size = train_agent_double_q(external_temp=np.random.uniform(-10, 10))
summer_rewards, summer_lengths, summer_q_table1, summer_q_table2, action_space_size = train_agent_double_q(external_temp=np.random.uniform(10, 30))

# The winter and summer average Q-tables were saved as CSV files
save_q_tables_as_csv(winter_q_table1, winter_q_table2, 'winter', action_space_size)
save_q_tables_as_csv(summer_q_table1, summer_q_table2, 'summer', action_space_size)

# The winter and summer average Q-tables were saved as images
save_q_tables_as_images(winter_q_table1, winter_q_table2, 'winter', action_space_size)
save_q_tables_as_images(summer_q_table1, summer_q_table2, 'summer', action_space_size)

# The resukts for the winter and summer results were plotted
plt.figure(figsize=(12, 6))

# Winter rewards were plotted using this part of the code
plt.subplot(2, 2, 1)
plt.plot(winter_rewards, label='Winter Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Winter: Episode Rewards Over Time')
plt.legend()

# Winter episode lengths were plotted using this part of the code
plt.subplot(2, 2, 2)
plt.plot(winter_lengths, label='Winter Episode Lengths', color='orange')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.title('Winter: Episode Lengths Over Time')
plt.legend()

# Summer rewards were plotted using this part of the code
plt.subplot(2, 2, 3)
plt.plot(summer_rewards, label='Summer Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Summer: Episode Rewards Over Time')
plt.legend()

# Summer episode lengths were plotted using this part of the code
plt.subplot(2, 2, 4)
plt.plot(summer_lengths, label='Summer Episode Lengths', color='orange')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.title('Summer: Episode Lengths Over Time')
plt.legend()

plt.tight_layout()
plt.show()

# This part of the code is for Printing the final Q-tables and aggregated results
output_directory = OUTPUT_DIR

print("Winter Training Completed. Final Q-table:")
print(winter_q_table1)
print(winter_q_table2)

print("Summer Training Completed. Final Q-table:")
print(summer_q_table1)
print(summer_q_table2)

print(f"Q-tables saved as CSV files and images in {output_directory}")

print("Winter Aggregated Results:")
print(f"Winter Average Reward: {np.mean(winter_rewards):.2f}")
print(f"Winter Max Reward: {np.max(winter_rewards):.2f}")
print(f"Winter Min Reward: {np.min(winter_rewards):.2f}")
print(f"Winter Average Episode Length: {np.mean(winter_lengths):.2f}")
print(f"Winter Max Episode Length: {np.max(winter_lengths):.2f}")
print(f"Winter Min Episode Length: {np.min(winter_lengths):.2f}")

print("Summer Aggregated Results:")
print(f"Summer Average Reward: {np.mean(summer_rewards):.2f}")
print(f"Summer Max Reward: {np.max(summer_rewards):.2f}")
print(f"Summer Min Reward: {np.min(summer_rewards):.2f}")
print(f"Summer Average Episode Length: {np.mean(summer_lengths):.2f}")
print(f"Summer Max Episode Length: {np.max(summer_lengths):.2f}")
print(f"Summer Min Episode Length: {np.min(summer_lengths):.2f}")
