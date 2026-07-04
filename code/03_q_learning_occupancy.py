"""Occupancy-aware Q-learning HVAC temperature-control experiment."""

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import os

#Defining the model learning parameters
class QLearning:
    def __init__(self, state_space_dims, action_space_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.1, replay_size=10000, batch_size=32):
        self.state_space_dims = state_space_dims
        self.action_space_size = action_space_size
        self.q_table = np.zeros((*state_space_dims, action_space_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory = deque(maxlen=replay_size)
        self.batch_size = batch_size

#In this part if the code, acyions were defined for when the room was occupied and non occupied, aggressive actions were limited for unoccupied rooms, and extreme actions were avoided
    def choose_action(self, state, occupancy_status):
        if random.uniform(0, 1) < self.epsilon:
            if occupancy_status == 0:  
                return random.choice([1, 2, 3])  
            else:
                return random.choice(range(self.action_space_size))  
        else:
            return np.argmax(self.q_table[state])  

    def update_q_table(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) >= self.batch_size:
            self.replay_experience()

    def replay_experience(self):
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in batch:
            best_next_action = np.argmax(self.q_table[next_state])
            td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
            td_error = td_target - self.q_table[state][action]
            self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self, occupancy_status):
        # Epsilon was made to decay more aggresively to give opportunity for more exploitation
        decay_rate = 0.998 if occupancy_status == 0 else self.epsilon_decay
        self.epsilon = max(self.min_epsilon, self.epsilon * decay_rate)

#Simulating the environment, by creating a bin for the temerature, occupancy status, and time. This was important as it was a cylic process that meant training for each hour of the day and the occupancy status varied according to the time of the day
class HVACEnvironment:
    def __init__(self, room_size, insulation, external_temp, hvac_power):
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
        action = max(0, min(action, 4))  # This line if the code was to ensure that the action was within range
        
        # The impact of the actions taken was limited when the room was unoccupied
        if self.occupancy_status == 0:  
            temp_change = (self.hvac_power * 0.5 * (action - 2) - self.insulation * (self.current_temp - self.external_temp))
        else:  # In occupied rooms the operations went on as normal
            temp_change = (self.hvac_power * (action - 2) - self.insulation * (self.current_temp - self.external_temp))
        
        self.current_temp += temp_change
        
        # Clarifying that there are two different target temperatures based on whether the room is occupied or unoccupied
        target_temp = 22 if self.occupancy_status == 1 else 18  
        deviation = abs(self.current_temp - target_temp)
        
        # This part of this code was added to avoid extreme negative rewards, and the positive rewards for moving close to the target temperature.
        if deviation < 0.5:
            reward = 10 
        elif deviation < 1.0:
            reward = 5   
        else:
            reward = -min(deviation, 20)  
        
        # The penalties for the unoccupied room is defined separately
        if self.occupancy_status == 0:
            reward -= abs(action - 2)  #
        
        
        reward = np.clip(reward, -50, 50)

        # This is the part of the code that was used to update the time of the day and occupancy status in this project.
        self.time_of_day = (self.time_of_day + 1) % 24
        self.occupancy_status = random.choice([0, 1])

        discrete_state = self.discretize_state()
        return discrete_state, reward

# Defininig the environment and agent
state_space_dims = (10, 24, 2) 
action_space_size = 5
agent = QLearning(state_space_dims, action_space_size, alpha=0.05, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, replay_size=10000, batch_size=64)

env = HVACEnvironment(room_size=50, insulation=0.1, external_temp=20, hvac_power=5)

episode_rewards = []
episode_lengths = []


occupied_rewards = []
unoccupied_rewards = []
occupied_lengths = []
unoccupied_lengths = []
occupied_indices = []  
unoccupied_indices = []  

# # The section below is the training loop for the agent
num_episodes = 1000000  # This training was done over 1,000,000 episodes
for episode in range(num_episodes):
    state = env.discretize_state()
    total_reward = 0
    length = 0
    done = False
    while not done:
        action = agent.choose_action(state, env.occupancy_status)
        next_state, reward = env.step(action)
        agent.update_q_table(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        length += 1
        if abs(env.current_temp - (22 if env.occupancy_status == 1 else 18)) < 0.5:  
            done = True
    agent.decay_epsilon(env.occupancy_status)  # The epsilon decay for this model kept getting adjusted based on whether there were occupants or not in the room 
    
    episode_rewards.append(total_reward)
    episode_lengths.append(length)
    
    # This is where the reward is being tracked
    if env.occupancy_status == 1:
        occupied_rewards.append(total_reward)
        occupied_lengths.append(length)
        occupied_indices.append(episode)  
    else:
        unoccupied_rewards.append(total_reward)
        unoccupied_lengths.append(length)
        unoccupied_indices.append(episode)

# The statistics of this model was cakcukate
def calculate_statistics(rewards, lengths):
    return {
        "average_reward": np.mean(rewards),
        "max_reward": np.max(rewards),
        "min_reward": np.min(rewards),
        "average_length": np.mean(lengths),
        "max_length": np.max(lengths),
        "min_length": np.min(lengths)
    }

# All the results fuction were printed out at the end of this training.
def plot_results(episodes, episode_rewards, episode_lengths, occupied_indices, occupied_rewards, unoccupied_indices, unoccupied_rewards):
    plt.figure(figsize=(10, 6))
    

    plt.plot(episodes, episode_rewards, label='Total Reward')
    plt.plot(episodes, episode_lengths, label='Episode Length')
    

    plt.plot(occupied_indices, occupied_rewards, label='Occupied Room Reward', linestyle='--')
    plt.plot(unoccupied_indices, unoccupied_rewards, label='Unoccupied Room Reward', linestyle='--')
    
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.xlabel('Episodes')
    plt.ylabel('Metric Value')
    plt.title('Performance Over Time')
    
    plt.tight_layout(rect=[0, 0, 0.8, 1])

    plt.show()


episodes = list(range(num_episodes))
plot_results(episodes, episode_rewards, episode_lengths, occupied_indices, occupied_rewards, unoccupied_indices, unoccupied_rewards)

print("Overall Statistics:", calculate_statistics(episode_rewards, episode_lengths))
print("Occupied Room Statistics:", calculate_statistics(occupied_rewards, occupied_lengths))
print("Unoccupied Room Statistics:", calculate_statistics(unoccupied_rewards, unoccupied_lengths))
