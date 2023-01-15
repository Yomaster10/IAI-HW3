from typing import List, Tuple, Dict
from termcolor import colored


class MDP:
    def __init__(self,
                 board: List[str],
                 terminal_states: List[Tuple[int]],
                 transition_function: Dict[str, Tuple[float]],
                 gamma: float):
        self.board = board
        self.num_row = len(board)
        self.num_col = len(board[0])
        self.terminal_states = terminal_states
        self.actions = {'UP': (-1, 0), 'DOWN': (1, 0), 'RIGHT': (0, 1), 'LEFT': (0, -1)}
        self.transition_function = transition_function
        self.gamma = gamma

    # returns the next step in the env
    def step(self, state, action):
        next_state = tuple(map(sum, zip(state, self.actions[action])))
        # collide with a wall
        if next_state[0] < 0 or next_state[1] < 0 or next_state[0] >= self.num_row or next_state[1] >= self.num_col or \
                self.board[next_state[0]][next_state[1]] == 'WALL':
            next_state = state
        return next_state

    def print_rewards(self):
        res = ""
        for r in range(self.num_row):
            res += "|"
            for c in range(self.num_col):
                val = self.board[r][c]
                if (r, c) in self.terminal_states:
                    res += " " + colored(val[:5].ljust(5), 'red') + " |"  # format
                elif self.board[r][c] == 'WALL':
                    res += " " + colored(val[:5].ljust(5), 'blue') + " |"  # format
                else:
                    res += " " + val[:5].ljust(5) + " |"  # format
            res += "\n"
        print(res)

    def print_utility(self, U: List[List[float]]):
        res = ""
        for r in range(self.num_row):
            res += "|"
            for c in range(self.num_col):
                if self.board[r][c] == 'WALL':
                    val = self.board[r][c]
                else:
                    val = str(U[r][c])
                if (r, c) in self.terminal_states:
                    res += " " + colored(str(round(float(val), 3))[:5].ljust(5), 'red') + " |"  # format
                elif self.board[r][c] == 'WALL':
                    res += " " + colored(val[:5].ljust(5), 'blue') + " |"  # format
                else:
                    res += " " + str(round(float(val), 3))[:5].ljust(5) + " |"  # format
            res += "\n"
        print(res)

    def print_policy(self, policy: List[List[float]]):
        res = ""
        for r in range(self.num_row):
            res += "|"
            for c in range(self.num_col):
                if self.board[r][c] == 'WALL' or (r, c) in self.terminal_states:
                    val = self.board[r][c]
                else:
                    val = str(policy[r][c])
                if (r, c) in self.terminal_states:
                    res += " " + colored(val[:5].ljust(5), 'red') + " |"  # format
                elif self.board[r][c] == 'WALL':
                    res += " " + colored(val[:5].ljust(5), 'blue') + " |"  # format
                else:
                    res += " " + val[:5].ljust(5) + " |"  # format
            res += "\n"
        print(res)
