import copy
import numpy as np
import random as rand
from soccer import player, environment
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

alpha = 0.3
initial_alpha = alpha
alpha_decay = 1
epsilon = 1.0
initial_epsilon =epsilon
decay = (0.001/initial_epsilon)**(1.0/1000000)

def plot(d):
    lists = sorted(d.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y)
    plt.title("Friend Q-learner")
    plt.xlabel('Simulation Iteration')
    plt.ylabel('Q-value difference')
    plt.ylim(0,0.5)
    plt.show()

def plot_no_zeros(d):
    lists = sorted(d.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y)
    plt.title("Friend Q-learner")
    plt.xlabel('Simulation Iteration')
    plt.ylabel('Q-value difference')
    plt.ylim(0, 0.5)
    plt.show()

num_states = 8*7*2
Q1 = np.full((num_states, 5, 5), 0.0, dtype=float)
Q2 = np.full((num_states, 5, 5), 0.0, dtype=float)

def couple_move(player_a, player_b, state, action_a, action_b):
    A_first = False
    done1 = False
    if np.random.rand() < 0.5:
        A_first = True
    if A_first:
        next_state, reward1, reward2, done1 = env.do_action(player_a, state, action_a, player_b)
        if not done1:
            next_state, reward1, reward2, done1 = env.do_action(player_b, next_state, action_b, player_a)
            return next_state, reward1, reward2, done1
        elif done1:
            next_state, rewardx1, rewardx2, donex = env.do_action(player_b, next_state, action_b, player_a)
            return next_state, reward1, reward2, done1
    else:
        next_state, reward1, reward2, done1 = env.do_action(player_b, state, action_b, player_a)
        if not done1:
            next_state, reward1, reward2, done1= env.do_action(player_a, next_state, action_a, player_b)
            return next_state, reward1, reward2, done1
        elif done1:
            next_state, rewardx1, rewardx2, donex = env.do_action(player_a, next_state, action_a, player_b)
            return next_state, reward1, reward2, done1

def get_action( player, state):
    if player.player_id =='A':
        return np.unravel_index(np.argmax(Q1[state]), Q1.shape)[1]
    if player.player_id =='B':
        return np.unravel_index(np.argmax(Q2[state]), Q2.shape)[2]

rows = 2
cols = 4
actions_a = []
actions_b = []
error_no_zero = {}
gamma = 0.9
done = False
time_step = 0
iteration_limit = 1000000
visits = 0
episode = 0
error = {}
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
        if np.random.rand() < epsilon :
            action_a = rand.randint(0,4)
            action_b = rand.randint(0,4)
        else:
            action_a = get_action(player_a, state)
            action_b = get_action(player_b, state)
        next_state, reward1, reward2, done = couple_move(player_a, player_b, state, action_a, action_b)
        previous_Q = Q1[initial_state, 2, 0].copy()
        if not done:
            Q1[state, action_a, action_b] = (1 - alpha) * Q1[state, action_a, action_b] + alpha * ((1 - gamma) * reward1 + gamma * np.max(Q1[next_state]))
            Q2[state, action_a, action_b] = (1 - alpha) * Q2[state, action_a, action_b] + alpha * ((1 - gamma) * reward2 + gamma * np.max(Q2[next_state]))
        elif done:
            Q1[state, action_a, action_b] = (1 - alpha) * Q1[state, action_a, action_b] + alpha * ((1 - gamma) * reward1)
            Q2[state, action_a, action_b] = (1 - alpha) * Q2[state, action_a, action_b] + alpha * ((1 - gamma) * reward2)
        if state ==initial_state and action_a ==2 and action_b ==0:
            visits +=1
        state = next_state
        current_Q = Q1[initial_state, 2, 0].copy()
        diff = abs(current_Q - previous_Q)
        error[time_step] = diff
        if diff!=0.0:
            error_no_zero[time_step] = diff
        time_step += 1
        alpha = max(0.001, alpha * alpha_decay)
        epsilon = max(0.001, decay * epsilon)
    if episode % 50000==0:
        print("EPISODE:", episode, "TIME STEP:", time_step,"EPSILON:", epsilon, "ALPHA", alpha)
    episode += 1

print("Q1[initial_state] values:")
print(Q1[initial_state])
print("Q2[initial_state] values:")
print(Q2[initial_state])

plot(error)
plot_no_zeros(error_no_zero)

# The testing phase can be uncommented to check how the agent behaves after its training period.
# (To uncomment, please remove the three single quotes before and after the testing phase block.)
'''print("TESTING PHASE")
test_episodes = 10
for i in range(test_episodes):
    player_a = player('A', 0, 2, False)
    player_b = player('B', 0, 1, True)
    state = env.initial_state
    initial_state = state
    actions = []
    done = False
    time_step = 0
    while not done and time_step < 10:
        action_a = get_action(player_a, state)
        action_b = get_action(player_b, state)
        A_first = False
        done = False
        if np.random.rand() < 0.5:
            A_first = True
        if A_first:
            actions.append('A' + str(action_a))
            actions.append('B' + str(action_b))
            next_state, reward1, reward2, done = env.do_action(player_a, state, action_a, player_b)
            next_state, reward1, reward2, done = env.do_action(player_b, next_state, action_b, player_a)
        else:
            actions.append('B' + str(action_b))
            actions.append('A' + str(action_a))
            next_state, reward1, reward2, done = env.do_action(player_b, state, action_b, player_a)
            next_state, reward1, reward2, done = env.do_action(player_a, next_state, action_a, player_b)
        state = next_state
        time_step+=1
    print("EPISODE:", i,"ACTIONS:", actions)'''
