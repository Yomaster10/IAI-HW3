from copy import deepcopy
import random
import numpy as np


def calculate_action_value(mdp, U, state, action):
    if state in mdp.terminal_states:
        return 0
    row = state[0]
    col = state[1]
    sum = 0
    probabilities = mdp.transition_function[action]
    possible_steps = []
    for dir in mdp.actions:
        possible_steps.append(mdp.step(state, dir))
    direction_index = 0
    for step in possible_steps:
        sum += probabilities[direction_index] * U[step[0]][step[1]]
        direction_index += 1
    return sum

def is_terminal_state(mdp, state):
    for terminal in mdp.terminal_states:
        if terminal[0] == state[0] and terminal[1] == state[1]:
            return True
    return False

def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    delta = float('inf')
    U_new = deepcopy(U_init)
    while delta >= (epsilon *(1- mdp.gamma)) / mdp.gamma:
        delta = 0
        U_init = deepcopy(U_new)
        for row in range(mdp.num_row):
            for col in range(mdp.num_col):
                state = [row, col]
                if mdp.board[row][col] == "WALL":
                    continue
                #find max action
                max_value = float('-inf')
                if is_terminal_state(mdp, state):
                    max_value = 0
                else:
                    for action in mdp.actions:
                        value = calculate_action_value(mdp, U_init, state, action)
                        if value > max_value:
                            max_value = value
            
                U_new[row][col] = float(mdp.board[row][col]) + (mdp.gamma * max_value)
                delta = max(delta, abs(U_new[row][col] - U_init[row][col]))
    
    return U_init



def get_policy(mdp, U):
    policy = np.empty(shape=(mdp.num_row, mdp.num_col), dtype='object')
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            if mdp.board[row][col] == 'WALL':
                continue
            #check policy for every state by looking at all options
            max_action = "NULL"
            max_value = float('-inf')
            state = [row, col]
            for action in mdp.actions:
                action_value = calculate_action_value(mdp, U, state, action)
                if action_value > max_value:
                    max_action = action
                    max_value = action_value
            policy[row][col] = max_action
    return policy


def q_learning(mdp, init_state, total_episodes=10000, max_steps=999, learning_rate=0.7, epsilon=1.0,
                      max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.8):
    # TODO:
    # Given the mdp and the Qlearning parameters:
    # total_episodes - number of episodes to run for the learning algorithm
    # max_steps - for each episode, the limit for the number of steps
    # learning_rate - the "learning rate" (alpha) for updating the table values
    # epsilon - the starting value of epsilon for the exploration-exploitation choosing of an action
    # max_epsilon - max value of the epsilon parameter
    # min_epsilon - min value of the epsilon parameter
    # decay_rate - exponential decay rate for exploration prob
    # init_state - the initial state to start each episode from
    # return: the Qtable learned by the algorithm
    #

    qtable = np.zeros((mdp.num_row * mdp.num_col, len(mdp.actions)))
    for episode in range(total_episodes):
        state = init_state
        step = 0
        done = False
        for step in range(max_steps):
            state_num = state[0] * len(qtable[0]) + state[1]
            exp_exp_tradeoff = random.uniform(0, 1)
            if exp_exp_tradeoff > epsilon:
                action = list(mdp.actions.values())[np.argmax(qtable[state_num,:])]
            else:
                action = random.choice(list(mdp.actions.values()))
            action = list(mdp.actions.keys())[list(mdp.actions.values()).index(action)]
            next_state = mdp.step(state, action)
            reward = float(mdp.board[next_state[0]][next_state[1]])
            done = (next_state in mdp.terminal_states)
            action_num = list(mdp.actions).index(action)
            next_state_num = next_state[0] * len(qtable[0]) + next_state[1]
            temp = np.max(qtable[next_state_num, :])
            qtable[state_num][action_num] = qtable[state_num][action_num] + (learning_rate * (reward + (mdp.gamma * np.max(qtable[next_state_num, :])) - qtable[state_num][action_num]))
            state = next_state
            if done:
                break
        epsilon = min_epsilon + ((max_epsilon - min_epsilon) * np.exp(-decay_rate * episode))
    return qtable



def q_table_policy_extraction(mdp, qtable):
    # TODO:
    # Given the mdp and the Qtable:
    # return: the policy corresponding to the Qtable
    #
    policy = np.empty(shape=(mdp.num_row, mdp.num_col), dtype='object')
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            if mdp.board[row][col] == 'WALL':
                continue
            state = row * mdp.num_col + col
            action = list(mdp.actions.keys())[np.argmax(qtable[state,:])]
            policy[row][col] = action
    return policy



# BONUS

def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
