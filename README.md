# Reinforcement Learning for Adaptive HVAC Temperature Control

Python implementations developed for my MSc thesis:

> **Reinforcement Learning for Adaptive Control Systems: Temperature Control Systems in an HVAC Unit**  
> MSc Advanced Control Systems, University of Salford, 2024

The project investigates reinforcement-learning approaches for temperature regulation in a simplified simulated HVAC environment. It includes tabular Q-learning, occupancy-aware control, Double Q-learning, Deep Q-learning and comparisons with PID control.

## Repository structure

```text
code/
  01_q_learning_basic.py
  02_q_learning_exploration_strategies.py
  03_q_learning_occupancy.py
  04_q_learning_occupancy_strategies.py
  05_double_q_learning_seasons.py
  06_double_q_learning_exploration_strategies.py
  07_rl_vs_pid_epsilon.py
  08_rl_vs_pid_softmax.py
  09_dqn_hvac.py

notebooks/
  Q_Learning_Occupancy_and_Double_Q_Learning.ipynb
  RL_vs_PID.ipynb

results/
  EXPERIMENTAL_RESULTS.md
  figures/

docs/
  Morakinyo_MSc_Thesis.pdf
```

The executed notebooks contain the project outputs and figures. The numbered Python files organise the experiments by method.

## Experiments

### 1. Basic Q-learning

A tabular Q-learning controller regulates a simplified room temperature around a 22°C target using ten temperature states and five discrete HVAC actions.

### 2. Exploration strategies

The project compares:

- epsilon-greedy;
- SoftMax action selection;
- Upper Confidence Bound (UCB).

### 3. Occupancy-aware control

The state space includes:

- temperature bin;
- hour of day;
- occupied or unoccupied status.

The target temperature is 22°C while occupied and 18°C while unoccupied.

### 4. Double Q-learning

Two Q-tables are used in seasonal simulations and in a separate comparison of epsilon-greedy, SoftMax and UCB exploration.

### 5. Reinforcement learning versus PID

Epsilon-greedy and SoftMax reinforcement-learning controllers are compared with a traditional PID controller using:

- final temperature error;
- response time;
- an internal action/control-effort calculation.

### 6. Deep Q-Network

The DQN experiment uses:

- a one-dimensional normalized state;
- five actions;
- two hidden layers with 32 neurons each;
- ReLU activation;
- Adam optimisation;
- experience replay;
- checkpoint saving;
- early stopping;
- up to 10,000 training episodes.

## Main simulation settings

- Room size: 50 m²
- Insulation coefficient: 0.1
- HVAC power parameter: 5
- Temperature range: 15–30°C
- Temperature bins: 10
- Action count: 5
- Occupied target: 22°C
- Unoccupied target: 18°C
- Time-of-day bins: 24 where applicable

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Running an experiment

From the repository root:

```bash
python code/01_q_learning_basic.py
```

The experiments can be computationally demanding. Some tabular experiments use up to 1,000,000 episodes, and the DQN uses up to 10,000 episodes.

## Results

The numerical outputs from the experiments are summarised in [`results/EXPERIMENTAL_RESULTS.md`](results/EXPERIMENTAL_RESULTS.md). The figures are available in [`results/figures`](results/figures).

## Scope

This study uses a simplified simulated room environment rather than a calibrated real-building HVAC installation. Results therefore describe the simulation and parameter settings used in the thesis.

The variable called `energy_consumption` in the RL-versus-PID scripts represents an internal control-effort proxy rather than measured electrical energy in kWh. It should be interpreted as an internal control-effort proxy rather than measured energy in kWh.

## Author

**Jesutomito Morakinyo**  
MSc Advanced Control Systems  
University of Salford
