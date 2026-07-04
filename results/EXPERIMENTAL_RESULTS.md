# Experimental Results

The values below are taken from the saved outputs of the executed project notebooks.

## Basic Q-learning

- Overall average reward: **7.16**
- Maximum reward: **184.80**
- Minimum reward: **-1546.05**
- Average episode length: **1.57**
- Maximum episode length: **146**
- Minimum episode length: **1**

## Q-learning exploration strategies

| Strategy | Average reward | Maximum reward | Minimum reward | Average episode length | Maximum length |
|---|---:|---:|---:|---:|---:|
| Epsilon-greedy | 7.1521 | 168.5805 | -1096.0383 | 1.5602 | 187 |
| SoftMax | 9.8628 | 23.2194 | -347.8835 | 4.1333 | 56 |
| UCB | 7.2677 | 10.0000 | -419.4345 | 3.5006 | 71 |

## Occupancy-aware Q-learning

| Condition | Average reward | Maximum reward | Minimum reward | Average episode length | Maximum length |
|---|---:|---:|---:|---:|---:|
| Overall | 14.6023 | 181.6009 | -3069.7759 | 4.6684 | 166 |
| Occupied | 14.3711 | 181.6009 | -2633.8652 | 4.5410 | 145 |
| Unoccupied | 15.2238 | 174.6765 | -3069.7759 | 5.0109 | 166 |

## Occupancy-aware exploration strategies

| Strategy | Average reward | Maximum reward | Minimum reward | Average episode length | Maximum length |
|---|---:|---:|---:|---:|---:|
| Epsilon-greedy | 5.5033 | 157.5730 | -3335.4483 | 5.6317 | 231 |
| SoftMax | 12.1578 | 139.3064 | -1805.3264 | 5.1344 | 147 |
| UCB | 7.9961 | 185.1609 | -2402.8235 | 5.2542 | 196 |

## Seasonal Double Q-learning

| Season | Average reward | Maximum reward | Minimum reward | Average episode length | Maximum length |
|---|---:|---:|---:|---:|---:|
| Winter | 12.49 | 167.49 | -6868.03 | 9.91 | 650 |
| Summer | 14.77 | 119.09 | -831.60 | 11.45 | 205 |

## Double Q-learning exploration strategies

| Strategy | Average reward | Maximum reward | Minimum reward | Average episode length | Maximum length |
|---|---:|---:|---:|---:|---:|
| Epsilon-greedy | 16.37 | 137.39 | -843.85 | 8.73 | 260 |
| SoftMax | 17.45 | 175.77 | -574.43 | 10.35 | 209 |
| UCB | 18.85 | 225.21 | -764.23 | 7.18 | 204 |

## Reinforcement learning versus PID

### Epsilon-greedy RL

| Metric | RL | PID |
|---|---:|---:|
| Average accuracy error | 0.25 | 0.32 |
| Average response time | 4.32 | 50.00 |
| Internal control-effort calculation | 15.42 | 7889.75 |

### SoftMax RL

| Metric | RL | PID |
|---|---:|---:|
| Average accuracy error | 0.24 | 0.32 |
| Average response time | 4.13 | 50.00 |
| Internal control-effort calculation | 12.75 | 7889.75 |

The quantity labelled `energy_consumption` is an internal action/control-effort calculation. It is not measured electrical energy in kWh.
