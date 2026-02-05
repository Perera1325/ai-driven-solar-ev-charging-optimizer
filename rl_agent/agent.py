import numpy as np
import random

class ChargingAgent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def get_state(self, battery, hour, day):
        return (round(battery, 1), hour, day)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1])

        if state not in self.q_table:
            self.q_table[state] = [0, 0]

        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0]

        old = self.q_table[state][action]
        future = max(self.q_table[next_state])

        self.q_table[state][action] = old + self.alpha * (reward + self.gamma * future - old)
