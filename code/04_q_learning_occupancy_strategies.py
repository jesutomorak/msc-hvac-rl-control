"""Exploration-strategy comparison for occupancy-aware Q-learning."""

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

class QLearning:
    def __init__(self, state_space_dims, action_space_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.1, replay_size=10000, batch_size=32):
        self.state_space_dims = state_space_dims
        self.action_space_size = action_space_size
        self.q_table = np.zeros((*state_space_dims, action_space_size))
        self.visit_counts = np.zeros((*state_space_dims, action_space_size))  # This line is used to track the visits for UCB
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory = deque(maxlen=replay_size)
        self.batch_size = batch_size

    def choose_action(self, state, occupancy_status):
        if random.uniform(0, 1) < self.epsilon:
            if occupancy_status == 0:  # In unoccupied rooms, the agent avoids taking aggressive actions, and explores normally in occupied rooms.
                return random.choice([1, 2, 3])  
            else:
                return random.choice(range(self.action_space_size))  
        else:
            return np.argmax(self.q_table[state])  # This line performs exploitation

    def choose_action_softmax(self, state):
        q_values = self.q_table[state]
        exp_q = np.exp(q_values - np.max(q_values))  # This line on code helps to avoid overflow
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(range(self.action_space_size), p=probs)

    def choose_action_ucb(self, state, step):
        if np.any(self.visit_counts[state] == 0):
            return np.random.choice(range(self.action_space_size))  # This line of code is used to explore the unvisited actions using UCB strategy

        total_visits = np.sum(self.visit_counts[state])  # This is where the record of the sum of all visited actions are kept and summed up
        action_visits = self.visit_counts[state]  # This is the line that stores the number of times actions are visited
        q_values = self.q_table[state]  

        ucb_values = q_values + np.sqrt(2 * np.log(total_visits + 1) / (action_visits + 1))
        return np.argmax(ucb_values)

    def update_q_table(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        self.visit_counts[state][action] += 1  # This line Updates the visit count for UCB
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
        decay_rate = 0.998 if occupancy_status == 0 else self.epsilon_decay
        self.epsilon = max(self.min_epsilon, self.epsilon * decay_rate)

class HVACEnvironment:
    def __init__(self, room_size, insulation, external_temp, hvac_power):
        self.room_size = room_size
        self.insulation = insulation
        self.external_temp = external_temp
        self.hvac_power = hvac_power
        self.current_temp = self.external_temp
        self.time_of_day = 0  
        self.occupancy_status = 1  # 1 in this line of code indicates that the room is occupied
        self.temp_bins = np.linspace(15, 30, 10)  # This line create bins for temperature ranging from 15 - 30 degrees
        self.time_bins = np.linspace(0, 23, 24)  # This line creates bins for each hour of the day

    def discretize_state(self):
        temp_state = np.digitize(self.current_temp, self.temp_bins) - 1
        time_state = np.digitize(self.time_of_day, self.time_bins) - 1
        return (temp_state, time_state, self.occupancy_status)

    def step(self, action):
        action = max(0, min(action, 4))  # Ensuring the action is within range
        
        # In this part of the code, the impact of the actions is limited
        if self.occupancy_status == 0:  # In unoccupied rooms large temperature shifts are avoided
            temp_change = (self.hvac_power * 0.5 * (action - 2) - self.insulation * (self.current_temp - self.external_temp))
        else:  # Occupied, normal operation
            temp_change = (self.hvac_power * (action - 2) - self.insulation * (self.current_temp - self.external_temp))
        
        self.current_temp += temp_change
        
        # This line sets separate penalty rules for when the room is occupied and unoccupied
        target_temp = 22 if self.occupancy_status == 1 else 18  
        deviation = abs(self.current_temp - target_temp)
        
        # This part if the code is used to set a lower cap on penalties
        if deviation < 0.5:
            reward = 10  
        elif deviation < 1.0:
            reward = 5   
        else:
            reward = -min(deviation, 20)  
        
        #This part of the code penalizes stronger HVAC actions
        if self.occupancy_status == 0:
            reward -= abs(action - 2)  
        
        # There was a cap put on extreme penalties
        reward = np.clip(reward, -50, 50)

        # The time of the day and the random occupancy changes are being updated using this part of the code
        self.time_of_day = (self.time_of_day + 1) % 24
        self.occupancy_status = random.choice([0, 1])

        discrete_state = self.discretize_state()
        return discrete_state, reward

def train_agent(agent, strategy, env, num_episodes):
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state = env.discretize_state()
        total_reward = 0
        length = 0
        done = False
        while not done:
            if strategy == 'epsilon':
                action = agent.choose_action(state, env.occupancy_status)
            elif strategy == 'softmax':
                action = agent.choose_action_softmax(state)
            elif strategy == 'ucb':
                action = agent.choose_action_ucb(state, episode)
                
            next_state, reward = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            length += 1
            if abs(env.current_temp - (22 if env.occupancy_status == 1 else 18)) < 0.5:
                done = True
        agent.decay_epsilon(env.occupancy_status)  
        
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

# Defining the environments and the agents
state_space_dims = (10, 24, 2)  # This represents the temperature bins, time bins, and occupancy status
action_space_size = 5
num_episodes = 10000

# Initializing the environments
env = HVACEnvironment(room_size=50, insulation=0.1, external_temp=20, hvac_power=5)

# Initializing the agents
agent_epsilon = QLearning(state_space_dims, action_space_size, alpha=0.05, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, replay_size=10000, batch_size=64)
agent_softmax = QLearning(state_space_dims, action_space_size, alpha=0.05, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, replay_size=10000, batch_size=64)
agent_ucb = QLearning(state_space_dims, action_space_size, alpha=0.05, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, replay_size=10000, batch_size=64)

# Training the agents
rewards_epsilon, lengths_epsilon = train_agent(agent_epsilon, 'epsilon', env, num_episodes)
rewards_softmax, lengths_softmax = train_agent(agent_softmax, 'softmax', env, num_episodes)
rewards_ucb, lengths_ucb = train_agent(agent_ucb, 'ucb', env, num_episodes)

# Calculating the statistics
stats_epsilon = calculate_statistics(rewards_epsilon, lengths_epsilon)
stats_softmax = calculate_statistics(rewards_softmax, lengths_softmax)
stats_ucb = calculate_statistics(rewards_ucb, lengths_ucb)

print("Epsilon-Greedy Statistics:", stats_epsilon)
print("Softmax Statistics:", stats_softmax)
print("UCB Statistics:", stats_ucb)

# Plotting the results
plot_comparison(rewards_epsilon, rewards_softmax, rewards_ucb, num_episodes)
