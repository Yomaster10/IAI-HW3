import argparse
import os
from mdp import MDP
from mdp_rl_implementation import value_iteration, get_policy, policy_evaluation, q_learning, q_table_policy_extraction


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)


def example_driver():
    """
    This is an example of a driver function, after implementing the functions
    in "mdp_rl_implementation.py" you will be able to run this code with no errors.
    """

    board = 'board'
    terminal_states = 'terminal_states'
    transition_function = 'transition_function'

    board_env = []
    with open(board, 'r') as f:
        for line in f.readlines():
            row = line[:-1].split(',')
            board_env.append(row)

    terminal_states_env = []
    with open(terminal_states, 'r') as f:
        for line in f.readlines():
            row = line[:-1].split(',')
            terminal_states_env.append(tuple(map(int, row)))

    transition_function_env = {}
    with open(transition_function, 'r') as f:
        for line in f.readlines():
            action, prob = line[:-1].split(':')
            prob = prob.split(',')
            transition_function_env[action] = tuple(map(float, prob))

    # initialising the env
    mdp = MDP(board=board_env,
              terminal_states=terminal_states_env,
              transition_function=transition_function_env,
              gamma=0.9)

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@ The board and rewards @@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    mdp.print_rewards()


    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@@@@ Value iteration @@@@@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    U = [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]

    print("\nInitial utility:")
    mdp.print_utility(U)
    print("\nFinal utility:")
    U_new = value_iteration(mdp, U)
    mdp.print_utility(U_new)
    print("\nFinal policy:")
    policy = get_policy(mdp, U_new)
    mdp.print_policy(policy)


    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@@@@@@@ QLearning @@@@@@@@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    print("\nBest Policy:")
    policy = q_table_policy_extraction(mdp, q_learning(mdp, (2, 0)))
    mdp.print_policy(policy)

    # Bonus
    """
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@@@@ Policy iteration @@@@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    print("\nPolicy evaluation:")
    U_eval = policy_evaluation(mdp, policy)
    mdp.print_utility(U_eval)

    policy = [['UP', 'UP', 'UP', 0],
              ['UP', 'WALL', 'UP', 0],
              ['UP', 'UP', 'UP', 'UP']]

    print("\nInitial policy:")
    mdp.print_policy(policy)
    print("\nFinal policy:")
    policy_new = policy_iteration(mdp, policy)
    mdp.print_policy(policy_new)

    print("Done!")
    """


if __name__ == '__main__':
    # run our example
    example_driver()
