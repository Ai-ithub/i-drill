# Drilling Simulation Environment

This package implements a drilling simulation environment following the OpenAI Gym interface. The environment simulates the physics and dynamics of drilling operations, allowing for the development and testing of drilling optimization algorithms.

## Installation

```bash
pip install -e .
```

## Environment Description

### State Space
The environment's state space consists of the following variables:
- Depth (m): Current depth of the wellbore
- Bit Wear (0-1): Wear factor of the drill bit
- ROP (m/hr): Rate of Penetration
- Torque (Nm): Drilling torque
- Pressure (Pa): Mud circulation pressure
- Vibrations: Axial, lateral, and torsional vibration levels

### Action Space
The action space consists of three continuous control variables:
1. Weight on Bit (WOB): 0-50,000 N
2. Rotary Speed (RPM): 0-200 rpm
3. Mud Flow Rate: 0-0.1 m³/s

### Reward Function
The reward function is designed to optimize drilling performance while considering operational constraints:
1. Positive reward for Rate of Penetration (ROP)
2. Penalties for:
   - Bit wear
   - Excessive vibrations
   - Operating costs (power consumption)

### Episode Termination
An episode ends when:
1. Maximum steps reached (default: 1000)
2. Bit wear exceeds 90%
3. Excessive vibrations detected

## Usage Example

```python
import gym
from drilling_env.drilling_env import DrillingEnv

# Create environment
env = DrillingEnv(max_episode_steps=1000)

# Reset environment
obs = env.reset()

# Run one episode
done = False
while not done:
    # Sample random action
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, done, info = env.step(action)
    
    # Render current state
    env.render()
```

## Physics Model

The environment uses a simplified physics model based on established drilling engineering principles:

1. Rate of Penetration (ROP):
   - Based on modified Bourgoyne-Young model
   - Considers WOB, RPM, and formation strength

2. Torque:
   - Function of WOB, bit diameter, and friction coefficient
   - Friction increases with bit wear

3. Pressure Loss:
   - Based on Hagen-Poiseuille equation
   - Considers mud properties and flow geometry

4. Bit Wear:
   - Progressive wear based on operating conditions
   - Affects drilling efficiency and vibrations

5. Vibrations:
   - Models axial, lateral, and torsional vibrations
   - Influenced by operating parameters and bit wear

## Validation

The environment's behavior has been validated against the historical drilling data provided in the `scripts/output_fastparquet/RIG_01.parquet` file. Key parameters and relationships have been calibrated to match observed drilling performance patterns.

## References

1. Bourgoyne Jr, A. T., & Young Jr, F. S. (1974). A multiple regression approach to optimal drilling and abnormal pressure detection.
2. Galle, E. M., & Woods, H. B. (1963). Best constant weight and rotary speed for rotary rock bits. 