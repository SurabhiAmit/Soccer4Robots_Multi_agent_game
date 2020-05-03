
import copy
import numpy as np
import random as rand
from soccer import player, environment
import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
from cvxopt.modeling import op
from cvxopt.modeling import variable
from cvxopt.solvers import options

alpha = 1.0
initial_alpha = alpha
alpha_decay = (0.001/initial_alpha)**(1.0/1000000)
epsilon = 1.0
initial_epsilon = epsilon
decay = 1.0

def plot(d):
    lists = sorted(d.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y, linewidth =0.4)
    plt.title("CE Q-learner")
    plt.xlabel('Simulation Iteration')
    plt.ylabel('Q-value difference')
    plt.ylim(0,0.5)
    plt.grid(True)
    plt.show()

num_states=8*7*2
Q1 = np.full((num_states, 5, 5), 1.0, dtype=float)
Q2 = np.full((num_states, 5, 5), -1.0, dtype=float)

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


def sum_of(listx):
    total =0
    for each in listx:
        total+=each
    return total

def normalize(prob_list):
    norm = 1 / sum_of(prob_list)
    return [norm * each for each in prob_list]

def get_action(listx):
    list_non_zero = (np.clip(listx, 0, 1)).tolist()
    list_prob = normalize((list_non_zero))
    act = np.random.choice(25, 1, p=list_prob)
    acts =[]
    if act!=0:
        acts.append((act)//5)
        acts.append(act%5)
    else:
        acts=[0,0]
    return acts

seed = 1
np.random.seed(seed)
rand.seed(seed)

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
abs_error = {}
env = environment(2, 4)
options['show_progress'] = False
while time_step < iteration_limit:
    player_a = player('A', 0, 2, False)
    player_b = player('B', 0, 1, True)
    state = env.initial_state
    initial_state = state
    done = False
    prob_listx=[]
    actions = []
    for i in range(5):
        for j in range(5):
            prob_listx.append(1.0/25)
    while not done:
        if np.random.rand() < epsilon:
            action_a = rand.randint(0, 4)
            action_b = rand.randint(0, 4)
        else:
            action_a = get_action(prob_listx)[0]
            action_b = get_action(prob_listx)[1]
        actions.append('A' + str(action_a))
        actions.append('B' + str(action_b))
        next_state, reward1, reward2, done = couple_move(player_a, player_b, state, action_a, action_b)
        previous_Q = Q1[initial_state, 2, 0].copy()
        if not done:
            obj = variable()
            joint_probs ={}
            constraints1 = list()

            #25 +1  constraints
            for i in range(5):
                for j in range(5):
                    joint_probs[(i,j)]=variable()
            for index, prob in joint_probs.items():
                constraints1.append((joint_probs[index] >= 0))
            total = sum(joint_probs.values())
            constraints1.append((total == 1))
            #1 constraint
            total_sum = 0
            for i in range(5):
                for j in range(5):
                    total_sum += float(Q1[next_state, i, j])* joint_probs[(i,j)]
                    total_sum += float(Q2[next_state, i, j]) *joint_probs[(i, j)]
            constraints1.append((obj - total_sum == 0))

            #20 constraints
            for a in range(5):
                exp_rew1 = 0
                for b in range(5):
                    exp_rew1 += joint_probs[(a,b)] * float(Q1[next_state, a, b])
                for other_a in range(5):
                    if other_a != a :
                        exp_rew2 = 0
                        for all_b in range(5):
                            exp_rew2 += joint_probs[(a,all_b)] * float(Q1[next_state, other_a, all_b])
                        constraints1.append((exp_rew1 >= exp_rew2))

            # 20 constraints
            for b in range(5):
                exp_rew1 = 0
                for a in range(5):
                    exp_rew1 += joint_probs[(a, b)] * float(Q2[next_state, a, b])
                for other_b in range(5):
                    if other_b != b:
                        exp_rew2 = 0
                        for all_a in range(5):
                            exp_rew2 += joint_probs[(all_a, b)] * float(Q2[next_state, all_a, other_b])
                        constraints1.append((exp_rew1 >= exp_rew2))

            lp1 = op(-obj, constraints1)
            lp1.solve()
            if next_state == initial_state and time_step > 999500:
                print("probabilities:")
                for key, value in joint_probs.items():
                    print (key,":", value.value[0])

            for i in range(5):
                for j in range(5):
                    prob_listx.append(joint_probs[(i, j)].value[0])

            v1_value, v2_value = 0, 0
            for i in range(5):
                for j in range(5):
                    v1_value += float(Q1[next_state, i, j])* joint_probs[(i,j)].value[0]
                    v2_value += float(Q2[next_state, i, j]) * joint_probs[(i, j)].value[0]

            Q1[state, action_a, action_b] = (1 - alpha) * Q1[state, action_a, action_b] + alpha * ((1 - gamma) * reward1 + gamma * v1_value)
            Q2[state, action_a, action_b] = (1 - alpha) * Q2[state, action_a, action_b] + alpha * ((1 - gamma) * reward2 + gamma * v2_value)
        elif done:
            Q1[state, action_a, action_b] = (1 - alpha) * Q1[state, action_a, action_b] + alpha * ((1 - gamma) * reward1)
            Q2[state, action_a, action_b] = (1 - alpha) * Q2[state, action_a, action_b] + alpha * ((1 - gamma) * reward2)
        state = next_state
        current_Q = Q1[initial_state, 2, 0].copy()
        diff = abs(current_Q - previous_Q)
        if diff >= 0.0 or time_step > 400000:
            error[time_step] = diff
        time_step += 1
        alpha = max(0.001, alpha * alpha_decay)
    if episode % 10000 == 0:
        print("EPISODE:", episode, "TIME STEP:", time_step, "EPSILON:", epsilon, "ALPHA", alpha)
    episode += 1

print("Q1[initial_state]:",Q1[initial_state])
print("Q2[initial_state]:",Q2[initial_state])

plot(error)
