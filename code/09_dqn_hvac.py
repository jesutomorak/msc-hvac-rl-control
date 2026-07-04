"""Deep Q-Network HVAC temperature-control experiment."""

# This model is the Deep Q-Network model, that was run to investigate an even advanced method of reinforcement learning for the purpose of optimizing the operation of an HVAC system in a simulated environment
# The agent trains over 10,000 episodes and there are saved checkpoints to make resuming the training easier in the case of any unexpected interruptions. There is also a mechanism for early stopping if the agent shows consistent signs of effective learning

import numpy as np
import random
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import psutil
import gc
import h5py


# This part of the code is to ensure the GPU memory is used efficiently
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# This part of the code initializes the state and action space sizes of the model, the memory for experience replay, and the learning parameters
class DQNAgent:
    def __init__(self, state_space_size, action_space_size, alpha=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, memory_size=20000, batch_size=128, checkpoint_dir=Path(__file__).resolve().parents[1] / "checkpoints"):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.model = self._build_model()

        # This part of the code is to makes sure the checkpoint directory exists
        if not os.path.exists(checkpoint_dir):
            print(f"Creating checkpoint directory at: {checkpoint_dir}")
            os.makedirs(checkpoint_dir)

        # This part of the code defines the checkpoint callback
        checkpoint_filepath = os.path.join(checkpoint_dir, 'dqn_model_episode_{epoch:02d}.weights.h5')
        print(f"Checkpoint file path: {checkpoint_filepath}")
        
        self.checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath, 
            save_weights_only=True,
            save_freq='epoch'
        )

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(self.state_space_size,)))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.action_space_size, activation='linear'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            # Saving the checkpoint during training
            self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[self.checkpoint_callback])
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

# The code below initializes environment parameters
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

    def normalize_state(self, state):
        return state / (len(self.temp_bins) - 1)  

    def step(self, action):
        action = max(0, min(action, 4))  # This part of the code ensures the action is within range
        temp_change = (self.hvac_power * (action - 2) - self.insulation * (self.current_temp - self.external_temp))
        self.current_temp += temp_change
        
        # This is the code that defines the reward function, and caps the penalties to avoid extreme negative rewards
        deviation = abs(self.current_temp - 22)
        if deviation < 0.5:
            reward = 10  
        elif deviation < 1.0:
            reward = 5   
        else:
            reward = -min(deviation, 50) 
        
        discrete_state = self.discretize_state(self.current_temp)
        return self.normalize_state(discrete_state), reward

# Defining the environment and agent
env = HVACEnvironment(room_size=50, insulation=0.1, external_temp=20, hvac_power=5)
agent = DQNAgent(state_space_size=1, action_space_size=5)

episode_rewards = []
episode_lengths = []
performance_threshold = 0.95  

# This is the Training loop of the model
max_steps_per_episode = 200  
for episode in range(10000):  
    state = np.reshape(env.discretize_state(env.current_temp), [1, 1])
    state = env.normalize_state(state)
    total_reward = 0
    length = 0
    done = False
    while not done and length < max_steps_per_episode:
        action = agent.choose_action(state)
        next_state, reward = env.step(action)
        next_state = np.reshape(next_state, [1, 1])
        done = abs(env.current_temp - 22) < 0.5  
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        length += 1
    agent.replay()  # Using previous experience from episodes to train the agent
    episode_rewards.append(total_reward)
    episode_lengths.append(length)

    # Checking for performance stability for early termination to avoid over training
    if episode % 100 == 0:
        recent_rewards = episode_rewards[-100:]
        recent_mean = np.mean(recent_rewards)
        print(f"Episode: {episode}, Recent Mean Reward: {recent_mean:.2f}, Memory Usage: {psutil.virtual_memory().percent}%")
        gc.collect()  

    if episode > 100:
        recent_rewards = episode_rewards[-100:]
        recent_mean = np.mean(recent_rewards)
        if recent_mean >= performance_threshold * 10: 
            print(f"Early stopping at episode {episode} with recent average reward: {recent_mean}")
            break

# Visualization
plt.figure(figsize=(12, 6))

# Plotting the rewards
plt.subplot(2, 1, 1)
plt.plot(episode_rewards, label='Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards Over Time')
plt.legend()

# Plotting the episode lengths
plt.subplot(2, 1, 2)
plt.plot(episode_lengths, label='Episode Lengths', color='orange')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.title('Episode Lengths Over Time')
plt.legend()

plt.tight_layout()
plt.show()

print("Training completed.")
