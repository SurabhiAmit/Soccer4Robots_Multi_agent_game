import copy
import numpy as np
import random

class player:

    def __init__(self, player_id, row_pos, col_pos, has_ball):
        self.row = row_pos
        self.col = col_pos
        self.player_id = player_id
        self.has_ball = has_ball

    def update_player_info(self, row, col, has_ball):
        self.row = row
        self.col = col
        self.has_ball = has_ball

    def update_row(self, row):
        self.row = row

    def update_col(self, col):
        self.col = col

    def update_ball(self, has_ball):
        self.has_ball = has_ball

class environment:

    def __init__(self, rows, columns):
        self.col = columns
        self.rows = rows
        self.actions = ['N', 'S', 'E', 'W', 'P'] # P for stay put
        self.verbose = False
        self.players = {}
        self.states = self.create_states()
        self.num_states = len(self.states.keys())
        self.initial_state= self.get_initial_state()

    def create_states(self):
        id = 0
        states={}
        ball = ['A', 'B']
        # each player can be in one of 8 states, but not together in one state
        for i in range(1,9):
            for j in range(1,9):
                if i!=j:
                    states[id] = 'A'+str(i*10+j)
                    id += 1
        for i in range(1,9):
            for j in range(1,9):
                if i !=j:
                    states[id] = 'B'+str(i*10+j)
                    id += 1
        return states

    def assign_player(self, player_id, current_player):
        self.players[player_id] = copy.copy(current_player)

    def update_goal_parameters(self, player_id, goal_col1, reward1, goal_col2, reward2):
        self.goals[player_id] = [[goal_col1, reward1],[goal_col2, reward2]]

    def exchange_ball(self, first_mover, second_mover):
        if second_mover.has_ball:
            second_mover.update_ball(False)
            first_mover.update_ball(True)

    def get_state_name(self, state_id):
        for id, state in self.states.items():
            if id == state_id:
                return state

    def get_player_state_id(self, row,col):
        if row == 0 and col == 0:
            state_id = 1
        elif row == 0 and col == 1:
            state_id = 2
        elif row == 0 and col == 2:
            state_id = 3
        elif row == 0 and col == 3:
            state_id = 4
        elif row ==1 and col == 0:
            state_id = 5
        elif row ==1 and col == 1:
            state_id = 6
        elif row ==1 and col == 2:
            state_id = 7
        elif row ==1 and col == 3:
            state_id = 8
        return state_id

    def exp_move(self, player, action):
        #action=[N=1, S=2, E=3, W=4, P=5]
        if player.row==0 and player.col==1:
            if action == 1:
                next_row = 0
                next_col = 1
            elif action == 2:
                next_row = 1
                next_col = 1
            elif action == 3:
                next_row = 0
                next_col = 2
            elif action == 4:
                next_row = 0
                next_col = 0
            elif action == 0:
                next_row = 0
                next_col = 1
        elif player.row == 0 and player.col == 2:
            if action == 1:
                next_row = 0
                next_col = 2
            elif action == 2:
                next_row = 1
                next_col = 2
            elif action == 3:
                next_row = 0
                next_col = 3
            elif action == 4:
                next_row = 0
                next_col = 1
            elif action == 0:
                next_row = 0
                next_col = 2
        elif player.row == 1 and player.col == 1:
            if action == 1:
                next_row = 0
                next_col = 1
            elif action == 2:
                next_row = 1
                next_col = 1
            elif action == 3:
                next_row = 1
                next_col = 2
            elif action == 4:
                next_row = 1
                next_col = 0
            elif action == 0:
                next_row = 1
                next_col = 1
        elif player.row == 1 and player.col == 2:
            if action == 1:
                next_row = 0
                next_col = 2
            elif action == 2:
                next_row = 1
                next_col = 2
            elif action == 3:
                next_row = 1
                next_col = 3
            elif action == 4:
                next_row = 1
                next_col = 1
            elif action == 0:
                next_row = 1
                next_col = 2
        if player.row==0 and player.col==0:
            if action == 1:
                next_row = 0
                next_col = 0
            elif action == 2:
                next_row = 1
                next_col = 0
            elif action == 3:
                next_row = 0
                next_col = 1
            elif action == 4:
                next_row = 0
                next_col = 0
            elif action == 0:
                next_row = 0
                next_col = 0
        if player.row==1 and player.col==0:
            if action == 1:
                next_row = 0
                next_col = 0
            elif action == 2:
                next_row = 1
                next_col = 0
            elif action == 3:
                next_row = 1
                next_col = 1
            elif action == 4:
                next_row = 1
                next_col = 0
            elif action == 0:
                next_row = 1
                next_col = 0
        if player.row==0 and player.col==3:
            if action == 1:
                next_row = 0
                next_col = 3
            elif action == 2:
                next_row = 1
                next_col = 3
            elif action == 3:
                next_row = 0
                next_col = 3
            elif action == 4:
                next_row = 0
                next_col = 2
            elif action == 0:
                next_row = 0
                next_col = 3
        if player.row==1 and player.col==3:
            if action == 1:
                next_row = 0
                next_col = 3
            elif action == 2:
                next_row = 1
                next_col = 3
            elif action == 3:
                next_row = 1
                next_col = 3
            elif action == 4:
                next_row = 1
                next_col = 2
            elif action == 0:
                next_row = 1
                next_col = 3
        return next_row, next_col

    def goal(self, player, next_col):
        if player.player_id == 'A' :
            if next_col == 0 and player.has_ball:
                return True
        if player.player_id == 'B':
            if next_col == 3 and player.has_ball:
                return True
        return False

    def reverse_goal(self, player, next_col):
        if player.player_id == 'A':
            if next_col == 3 and player.has_ball:
                return True
        if player.player_id == 'B':
            if next_col == 0 and player.has_ball:
                return True
        return False

    def do_action(self, player, state_id, action, opponent):
        reward1 =0
        reward2 =0
        done=False
        state = self.states[state_id]
        next_row,next_col = self.exp_move(player, action)
        if opponent.row == next_row and opponent.col == next_col:
            self.exchange_ball(opponent,player)
            player_next_state_id = self.get_player_state_id(player.row, player.col)
        else:
            player_next_state_id = self.get_player_state_id(next_row,next_col)
            player.row = next_row
            player.col = next_col
        if player.player_id == 'A':
            if player.has_ball:
                next_state = 'A'+str(player_next_state_id)+state[-1:]
            elif not player.has_ball:
                next_state = 'B'+str(player_next_state_id)+state[-1:]
        elif player.player_id == 'B':
            if player.has_ball:
                next_state = 'B'+state[-2:-1]+str(player_next_state_id)
            elif not player.has_ball:
                next_state = 'A'+state[-2:-1]+str(player_next_state_id)
        next_state_id =-1
        for id1, state1 in self.states.items():
            if state1 == next_state:
                next_state_id = id1
        if next_state_id ==-1:
            print("ERROR!! NEXT STATE NOT FOUND")
            print("NEXT STATE WHICH HAS NO ID IS",  next_state)
        if player.player_id == 'A':
            if self.goal(player, player.col):
                reward1 = 100
                reward2 = -100
                done = True
            elif self.reverse_goal(player, player.col):
                reward1 = -100
                reward2 = 100
                done = True
        if player.player_id == 'B':
            if self.goal(player, player.col):
                reward2 = 100
                reward1 = -100
                done = True
            elif self.reverse_goal(player, player.col):
                reward2 = -100
                reward1 = 100
                done = True
        return next_state_id, reward1, reward2, done

    def get_initial_state(self):
        for id, state in self.states.items():
            if state == "B32":
                return id


