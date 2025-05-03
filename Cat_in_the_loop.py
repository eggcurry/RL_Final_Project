### CAT-IN-THE-LOOP ###
## # Written by Crista Falk and Vanalata Bulusu

## # Course project for RL by Jivko Sinapov, Fall 2025. 
## Building a cat toy that your cat is actually engaged by. The toy will learn your cat's behavior and avoid being caught.
## This code implements a simple Q-learning agent that interacts with an environment using infrared and ultrasound sensors.

## import modules
# for math
import numpy as np
# for random number generation
import random

## agent
class QLearningAgent:
    def __init__(self, actions, state_space, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions  # List of possible actions
        self.state_space = state_space  # State space representation
        self.q_table = np.zeros(state_space + (len(actions),))  # Initialize Q-table
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:
            return self.actions[np.argmax(self.q_table[state])]  # Exploit

    def update_q_value(self, state, action, reward, next_state):
        action_index = self.actions.index(action)
        best_next_action = np.max(self.q_table[next_state])
        self.q_table[state][action_index] += self.alpha * (
            reward + self.gamma * best_next_action - self.q_table[state][action_index]
        )

## environment interaction
def calculate_state(infrared_input, ultrasound_input, thresholds):
    # Discretize sensor inputs into states
    infrared_state = int(infrared_input < thresholds['infrared'])
    ultrasound_state = int(ultrasound_input < thresholds['ultrasound'])
    return (infrared_state, ultrasound_state)

def get_reward(state):
    # Define reward function
    if state == (1, 1):  # Both sensors detect obstacle
        return -10
    elif state == (1, 0) or state == (0, 1):  # One sensor detects obstacle
        return -5
    else:  # No obstacle detected
        return 1

## Running the agent
def main():
    actions = ['forward', 'left', 'right', 'backward', 'stop']
    state_space = (2, 2)  # Infrared and ultrasound states (binary), velocity
    agent = QLearningAgent(actions, state_space)

    thresholds = {'infrared': 10, 'ultrasound': 15}  # Example thresholds for sensors

    for episode in range(1000):  # Training loop
        infrared_input = np.random.uniform(0, 20)  # Simulated infrared input
        ultrasound_input = np.random.uniform(0, 20)  # Simulated ultrasound input

        state = calculate_state(infrared_input, ultrasound_input, thresholds)
        action = agent.choose_action(state)

        # Simulate environment response
        next_infrared_input = np.random.uniform(0, 20)
        next_ultrasound_input = np.random.uniform(0, 20)
        next_state = calculate_state(next_infrared_input, next_ultrasound_input, thresholds)
        reward = get_reward(state)

        agent.update_q_value(state, action, reward, next_state)

        if episode % 100 == 0:
            print(f"Episode {episode}: State {state}, Action {action}, Reward {reward}")

if __name__ == "__main__":
    main()
