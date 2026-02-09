# MSc Thesis Project — Reinforcement Learning for HVAC Temperature Control (RL vs PID)

**Focus:** Adaptive temperature control in a simulated HVAC room environment using Reinforcement Learning (Q-learning / Double Q-learning) and comparison against a traditional PID controller.

**Tools:** MATLAB, Simulink and Python

## Problem
HVAC controllers must keep indoor temperature comfortable while avoiding slow responses and inefficient control actions. Classical controllers like PID can work well but may struggle in dynamic conditions (e.g., changing occupancy-driven setpoints and varying external temperature).

This project investigates whether a Reinforcement Learning (RL) controller can learn an effective control policy for temperature regulation in a simulated HVAC environment, and how it compares to a PID baseline under the same conditions.

## Data / Simulation Environment
Because real HVAC experiments are costly and slow, the work is evaluated in a **simulation environment** representing room temperature dynamics.

Key environment design:
- **Temperature discretization:** 10 bins covering **15°C to 30°C**
- **Time-of-day discretization:** **24 bins** (0–23 hours)
- **Occupancy:** 2 bins (**occupied / unoccupied**)
- **Actions:** 5 discrete actions (0–4) representing different HVAC adjustment levels (heating → cooling)
- **Target temperature (setpoint) changes with occupancy:**
  - **22°C when occupied**
  - **18°C when unoccupied**
- **Dynamics / realism:** occupancy status changes (randomly), time-of-day advances, and external temperature varies across time (including winter/summer comparisons in Double Q-learning)
- **Reward design:** higher reward when the room temperature deviation from target is small:
  - high reward when deviation < **0.5°C**
  - medium reward when deviation between **0.5°C and 1°C**
  - negative reward when deviation > **1°C**

## Method (RL vs PID)

### Baseline: PID Controller
A traditional PID controller is used as the benchmark. It adjusts HVAC control based on temperature error to the setpoint. This serves as the reference for response time and energy consumption in the dynamic environment.

### Reinforcement Learning Controller (Q-learning / Double Q-learning)
RL treats temperature regulation as a sequential decision-making task:
- **State:** (temperature bin, time bin, occupancy)
- **Action:** one of 5 HVAC adjustment levels
- **Reward:** based on deviation from target temperature and efficiency-related behaviour

Exploration strategies tested (Q-learning):
- **Epsilon-Greedy**
- **SoftMax**
- **UCB (Upper Confidence Bound)**

A Double Q-learning model is also evaluated (including winter/summer scenarios) to reduce overestimation bias and improve stability.

## Evaluation Metrics (as used in the thesis)
Instead of classic control metrics (overshoot/settling time/IAE), the evaluation in this project focuses on:

1) **Response Time**  
How quickly the controller reaches/maintains the target region (in RL setups, episodes can terminate when within ±0.5°C or when the max step limit is reached).

2) **Accuracy (thesis-defined)**  
An accuracy measure computed in the thesis to quantify tracking performance relative to the target temperature.

3) **Energy Consumption**  
A numerical energy usage measure produced by the simulation environment (efficiency proxy).

4) **RL Training Metrics**
- **Episode reward**
- **Episode length** (steps to completion / termination)

## Robustness Tests (realistic variability)
Robustness is tested by introducing dynamism into the environment:
- **Random initial temperature / starting conditions**
- **Occupancy-driven setpoint switching (22°C ↔ 18°C)**
- **Time-of-day behaviour (24-hour bins influencing dynamics)**
- **External temperature variation**
- **Seasonal comparisons (winter vs summer) in Double Q-learning**

## Results Snapshot
In the dynamic environment, RL achieves much faster response and dramatically lower energy use than PID:

| Comparison | Controller | Accuracy (thesis metric) | Response Time | Energy Consumption |
|---|---:|---:|---:|---:|
| RL vs PID (Epsilon-Greedy) | RL  | 0.25 | 4.32 | 15.42 |
| RL vs PID (Epsilon-Greedy) | PID | 0.32 | 50.00 | 7889.75 |
| RL vs PID (SoftMax) | RL  | 0.24 | 4.13 | 12.75 |
| RL vs PID (SoftMax) | PID | 0.32 | 50.00 | 7889.75 |

**Interpretation (plain English):**
- RL reaches the target region far quicker (≈ 4 steps vs PID hitting max/slow response at 50).
- PID is extremely energy-inefficient in this unstable/dynamic environment, while RL remains low-energy.

### Exploration Strategy Comparisons (Q-learning & Double Q-learning)
**Q-learning (Table 4.1):** SoftMax has the best average reward (9.86) vs Epsilon-Greedy (7.15) and UCB (7.27).  
**Q-learning with occupancy randomness (Table 4.2):** SoftMax again gives the best average reward (12.16).  
**Double Q-learning (Table 4.3):** UCB performs best on average reward (18.85) and is more efficient on average episode length (7.18).

Full thesis report: [Download PDF](report/Dissertation_2024_Jesutomito.docx)

## Contact
If you’d like to discuss the project or see implementation details, feel free to reach out:
- Email: jesutomorak@gmail.com
- LinkedIn: https://www.linkedin.com/in/jesutomito-morakinyo-83b97a123
