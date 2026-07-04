"""Basic Q-learning HVAC temperature-control experiment."""

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

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.q_table.shape[1]))  # Line of action for exploration.
        else:
            return np.argmax(self.q_table[state])  # Line of action for exploitation.

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

#Simulating the environment
class HVACEnvironment:
    def __init__(self, room_size, insulation, external_temp, hvac_power):
        self.room_size = room_size
        self.insulation = insulation
        self.external_temp = external_temp
        self.hvac_power = hvac_power
        self.current_temp = self.external_temp
        self.temp_bins = np.linspace(15, 30, 10)  # The temperature was separated into bins of 10 using this line

    def discretize_state(self, temp):
        if temp < self.temp_bins[0]:
            return 0
        elif temp > self.temp_bins[-1]:
            return len(self.temp_bins) - 1
        return np.digitize(temp, self.temp_bins) - 1 

    def step(self, action):
        action = max(0, min(action, 4))  # This line is for making sure the action is within range
        temp_change = (self.hvac_power * (action - 2) - self.insulation * (self.current_temp - self.external_temp))
        self.current_temp += temp_change
        
        # Reward function is being updated, giving a bonus for getting close to the target temperature, medium reward for a deviation greater than 0.5 but less than 1 degree, and a negative reward proportional to the deviation when the deviation is greater than 1 degree
        deviation = abs(self.current_temp - 22)
        if deviation < 0.5:
            reward = 10 
        elif deviation < 1.0:
            reward = 5  
        else:
            reward = -min(deviation, 20) 
        
        discrete_state = self.discretize_state(self.current_temp)
        return discrete_state, reward

# Defining the environment and the agent
env = HVACEnvironment(room_size=50, insulation=0.1, external_temp=20, hvac_power=5)
agent = QLearning(state_space_size=10, action_space_size=5, alpha=0.05, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01)

episode_rewards = []
episode_lengths = []

# The section below is the training loop for the agent.
num_episodes = 1000000  # 1000000 episodes was used to train this agent
for episode in range(num_episodes):
    state = env.discretize_state(env.current_temp)
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
        if abs(env.current_temp - 22) < 0.5:  
            done = True
    agent.decay_epsilon()  # Epsilon was set to decay over time with this line of code
    episode_rewards.append(total_reward)
    episode_lengths.append(length)

# This section was used to visualize the results obtained from this model
plt.figure(figsize=(12, 6))

# The code for plotting the episode reward result
plt.subplot(2, 1, 1)
plt.plot(episode_rewards, label='Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards Over Time')
plt.legend()

# The code for plotting the episode length result
plt.subplot(2, 1, 2)
plt.plot(episode_lengths, label='Episode Lengths', color='orange')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.title('Episode Lengths Over Time')
plt.legend()

plt.tight_layout()
plt.show()

# The code for saving the q-table from this Q-learning model
q_table_image_path = OUTPUT_DIR / "q_table.png"
plt.figure(figsize=(10, 8))
sns.heatmap(agent.q_table, annot=True, fmt=".2f", cmap='viridis')
plt.title('Q-Table Heatmap')
plt.xlabel('Actions')
plt.ylabel('States')
plt.savefig(q_table_image_path)
plt.close()

print(f"Q-table saved as an image at {q_table_image_path}")

# The results are being printed using this
print("Training completed. Final Q-table:")
print(agent.q_table)

print("Aggregated Results:")
print(f"Overall Average Reward: {np.mean(episode_rewards):.2f}")
print(f"Max Reward: {np.max(episode_rewards):.2f}")
print(f"Min Reward: {np.min(episode_rewards):.2f}")
print(f"Average Episode Length: {np.mean(episode_lengths):.2f}")
print(f"Max Episode Length: {np.max(episode_lengths):.2f}")
print(f"Min Episode Length: {np.min(episode_lengths):.2f}")
