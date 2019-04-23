from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product
import ast


class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        
        '''
        Comment from Anugraha Sinha:
        We are visualizing the game in following fashion (The number in boxes are the idx values of "self.states"
         ____________________
        |      |      |      |
        |   0  |   1  |   2  |
        |______|______|______|
        |      |      |      |
        |   3  |   4  |   5  |
        |______|______|______|
        |      |      |      |
        |   6  |   7  |   8  |
        |______|______|______|
        
        We use a strict way of checking the sum as 15, because np.diagonal provides only primary diagonal (top-left -> bottom-right)
        However, we would also want to check (top-right -> bottom-left), hence we do not follow the simple np.sum methods, but define the indexes individually
        '''
        horizontals = [[0,1,2],[3,4,5],[6,7,8]]
        verticals = [[0,3,6],[1,4,7],[2,5,8]]
        diagonals = [[0,4,8],[2,4,6]]
        
        hor_ans = list(filter(lambda x : np.sum(np.array(curr_state)[x]) == 15,horizontals))
        ver_ans = list(filter(lambda x : np.sum(np.array(curr_state)[x]) == 15,verticals))
        diag_ans = list(filter(lambda x : np.sum(np.array(curr_state)[x]) == 15,diagonals))
        
        if len(hor_ans) != 0 or len(ver_ans) != 0 or len(diag_ans) != 0:
            return True
        else:
            return False

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 != 0]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 == 0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [9, 7]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        
        '''
        Comment from Anugraha Sinha:
        As stated in the comment above, we will consider curr_action as a list which means
        curr_action = [position to be added, number to be added]
        
        Note : 
        1. "Position" is in python addressing system
        2. There is a problem with the example given
           action space returns (position,number) type of data, however, the example given above states opposite.
           We are going with (position,number) type of curr_action
        '''
        new_state = curr_state.copy()                  ## Important : Good if we copy things in order to avoid reference to same memory location #
        new_state[curr_action[0]] = curr_action[1]
        return new_state

    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [9, 7]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        
        # update state with agent's action #
        new_state = self.state_transition(curr_state,curr_action)
        
        # check the status (win/loss/tie) after agent's move
        chk_terminal,message = self.is_terminal(new_state)
        
        if chk_terminal:
            # If we reached terminal state, display message
            #print("After agent's move, reached terminal state : %s" %(message))
            # If it was win, reward will be '10'
            if message == "Win":
                reward = 10
                msg = "Agent Won"
            # if it was tie, reward will '0'
            else:
                reward = 0
                msg = "Tie"
            return new_state,reward,chk_terminal,msg
        else:
            # It is not a terminal state after agent's move #
            # execute move for environment #
            
            # Environment's next action would be random selection from available action space #
            env_action = random.choice([ac for itr,ac in enumerate(self.action_space(new_state)[1])])
            
            # execute environment action #
            new_state = self.state_transition(new_state,env_action)
            
            # we should check if environment's action resulted in terminal state #
            chk_terminal_2, message_2 = self.is_terminal(new_state)
            
            if chk_terminal_2:
                # if we reached a terminal state #
                #print("After agent's action of : %s, environment took action as : %s. This led to terminal state : %s" %(str(curr_action),str(env_action),message_2))
                # if it was win by environment, then reward should be "-10"
                if message_2 == "Win":
                    reward = -10
                    msg = "Env Won"
                # if it was tie, reward will be '0'
                else:
                    reward = 0
                    msg = "Tie"
                return new_state,reward,chk_terminal_2,msg
            else:
                # even after environment action, we have "resume", and hence reward = "-1"
                reward = -1
                msg = "Continue"
                return new_state, reward, chk_terminal_2,msg

    def game_step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [9, 7]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        
        # update state with agent's action #
        new_state = self.state_transition(curr_state,curr_action)
        print("Agent Action : Position = %d, number = %d" %(curr_action[0],curr_action[1]))
        print("After agent action, state")
        print(self.print_board(new_state))
        
        # check the status (win/loss/tie) after agent's move
        chk_terminal,message = self.is_terminal(new_state)
        
        if chk_terminal:
            # If we reached terminal state, display message
            #print("After agent's move, reached terminal state : %s" %(message))
            # If it was win, reward will be '10'
            if message == "Win":
                reward = 10
                msg = "Agent Won"
            # if it was tie, reward will '0'
            else:
                reward = 0
                msg = "Tie"
            return new_state,reward,chk_terminal,msg
        else:
            # It is not a terminal state after agent's move #
            # execute move for environment #
            
            # Get user input for action #
            #env_action = random.choice([ac for itr,ac in enumerate(self.action_space(new_state)[1])])
            user_action = ast.literal_eval(input("Give user action (position,number):"))
            
            # execute environment action #
            new_state = self.state_transition(new_state,user_action)
            print("After user input : Position = %d, number = %d" %(user_action[0],user_action[1]))
            print(self.print_board(new_state))
            
            # we should check if environment's action resulted in terminal state #
            chk_terminal_2, message_2 = self.is_terminal(new_state)
            
            if chk_terminal_2:
                # if we reached a terminal state #
                #print("After agent's action of : %s, environment took action as : %s. This led to terminal state : %s" %(str(curr_action),str(env_action),message_2))
                # if it was win by environment, then reward should be "-10"
                if message_2 == "Win":
                    reward = -10
                    msg = "User Won"
                # if it was tie, reward will be '0'
                else:
                    reward = 0
                    msg = "Tie"
                return new_state,reward,chk_terminal_2,msg
            else:
                # even after environment action, we have "resume", and hence reward = "-1"
                reward = -1
                msg = "Continue"
                return new_state, reward, chk_terminal_2,msg

    def reset(self):
        return self.state
        
    def print_board(self,curr_state):
        printable_curr_state = ["X" if np.isnan(x) else str(x) for x in curr_state]
        return "    %s\n    %s\n    %s" %(str(np.array(printable_curr_state).reshape(3,3)[0,:]),str(np.array(printable_curr_state).reshape(3,3)[1,:]),str(np.array(printable_curr_state).reshape(3,3)[2,:]))
