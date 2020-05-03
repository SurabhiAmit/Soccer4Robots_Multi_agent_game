import copy
import numpy as np
import random as rand
from soccer import player, environment
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

alpha = 0.6
initial_alpha = alpha
alpha_decay = 0.999996
epsilon = 0.9
initial_epsilon = epsilon
decay =0.999995

def plot(d):
    lists = sorted(d.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y, linewidth =0.4)
    plt.title("Regular Q-learner")
    plt.xlabel('Simulation Iteration')
    plt.ylabel('Q-value difference')
    plt.ylim(0,0.5)
    plt.show()

num_states=8*7*2
Q1 = np.full((num_states, 5), 0.0, dtype=float)
Q2 = np.full((num_states, 5), 0.0, dtype=float)

def get_action( player, state):
    if player.player_id =='A':
        return np.argmax(Q1[state, :])
    if player.player_id =='B':
        return np.argmax(Q2[state,:])

def couple_move(player_a, player_b, state, action_a, action_b):
    A_first = False
    done = False
    if np.random.rand() < 0.5:
        A_first = True
    if A_first:
        next_state, reward1, reward2, done = env.do_action(player_a, state, action_a, player_b)
        if not done:
            next_state, reward1, reward2, done = env.do_action(player_b, next_state, action_b, player_a)
            return next_state, reward1, reward2, done
        elif done:
            next_state, rewardx1, rewardx2, donex = env.do_action(player_b, next_state, action_b, player_a)
            return next_state, reward1, reward2, done
    else:
        next_state, reward1, reward2, done = env.do_action(player_b, state, action_b, player_a)
        if not done:
            next_state, reward1, reward2, done = env.do_action(player_a, next_state, action_a, player_b)
            return next_state, reward1, reward2, done
        elif done:
            next_state, rewardx1, rewardx2, donex = env.do_action(player_a, next_state, action_a, player_b)
            return next_state, reward1, reward2, done

rows = 2
cols = 4
actions_a = []
actions_b = []

gamma = 0.9
done = False
time_step = 0
iteration_limit = 1000000
visits = 0
episode = 0
error = {}
error_no_zero={}
abs_error = {}
env = environment(2, 4)
while time_step < iteration_limit:
    player_a = player('A', 0, 2, False)
    player_b = player('B', 0, 1, True)
    state = env.initial_state
    initial_state = state
    done = False
    actions = []
    while not done:
        if np.random.rand() < epsilon or episode < 200:
            action_a = rand.randint(0,4)
            action_b = rand.randint(0,4)
        else:
            action_a = get_action(player_a, state)
            action_b = get_action(player_b, state)
        actions.append('A' + str(action_a))
        actions.append('B' + str(action_b))
        next_state, reward1, reward2, done = couple_move(player_a, player_b, state, action_a, action_b)
        previous_Q = Q1[initial_state, 2].copy()
        if not done:
            Q1[state, action_a] = (1 - alpha) * Q1[state, action_a] + alpha * ((1 - gamma) * reward1 + gamma * np.max(Q1[next_state]))
            Q2[state, action_b] = (1 - alpha) * Q2[state, action_b] + alpha * ((1 - gamma) * reward2 + gamma * np.max(Q2[next_state]))
        elif done:
            Q1[state, action_a] = (1 - alpha) * Q1[state, action_a] + alpha * ((1 - gamma) * reward1)
            Q2[state, action_b] = (1 - alpha) * Q2[state, action_b] + alpha * ((1 - gamma) * reward2)
        state = next_state
        current_Q = Q1[initial_state, 2].copy()
        diff = abs(current_Q - previous_Q)
        if diff!=0.0 :
            error_no_zero[time_step] = diff
        error[time_step] = diff
        time_step += 1
        alpha = max(0.001, alpha * alpha_decay)
        epsilon = max(0.001, decay * epsilon)
    if episode % 50000==0:
        print("EPISODE:", episode,"TIME STEP:",time_step,"EPSILON:", epsilon, "ALPHA", alpha)
    episode += 1

print("Q1[INITIAL_STATE]", Q1[initial_state])
print("Q2[initial_state]:",Q2[initial_state])

plot(error_no_zero)


