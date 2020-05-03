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
    plt.title("Foe Q-Learner")
    plt.xlabel('Simulation Iteration')
    plt.ylabel('Q-value difference')
    plt.ylim(0, 0.5)
    plt.grid(True)
    plt.show()

num_states=8*7*2
Q1 = np.full((num_states, 5, 5), 1.0, dtype=float)
Q2 = np.full((num_states, 5, 5), -1.0, dtype=float)

def couple_move(player_a1, player_b1, state1, action_a1, action_b1):
    A_first = False
    done = False
    if np.random.rand() < 0.5:
        A_first = True
    if A_first:
        next_state, reward1, reward2, done = env.do_action(player_a1, state1, action_a1, player_b1)
        if not done:
            next_state, reward1, reward2, done = env.do_action(player_b1, next_state, action_b1, player_a1)
            return next_state, reward1, reward2, done
        elif done:
            next_state, rewardx1, rewardx2, donex = env.do_action(player_b1, next_state, action_b1, player_a1)
            return next_state, reward1, reward2, done
    else:
        next_state, reward1, reward2, done = env.do_action(player_b1, state1, action_b1, player_a1)
        if not done:
            next_state, reward1, reward2, done = env.do_action(player_a1, next_state, action_a1, player_b1)
            return next_state, reward1, reward2, done
        elif done:
            next_state, rewardx1, rewardx2, donex = env.do_action(player_a1, next_state, action_a1, player_b1)
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
    act = np.random.choice(5, 1, p=list_prob)
    return act

seed =1
np.random.seed(seed)
rand.seed(seed)
rows = 2
cols = 4
actions_a = []
actions_b = []
neg=0
gamma = 0.9
done = False
time_step = 0
iteration_limit = 1000000
visits = 0
episode = 0
error = {}
env = environment(2, 4)
options['show_progress'] = False
list1 = [0.2, 0.2, 0.2, 0.2, 0.2]
list2 = [0.2, 0.2, 0.2, 0.2, 0.2]
while time_step < iteration_limit:
    player_a = player('A', 0, 2, False)
    player_b = player('B', 0, 1, True)
    state = env.initial_state
    initial_state = state
    done = False
    actions = []
    while not done:
        if np.random.rand() < epsilon:
            action_a = rand.randint(0, 4)
            action_b = rand.randint(0, 4)
        else:
            action_a = get_action(list1)
            action_b = get_action(list2)
        actions.append('A' + str(action_a))
        actions.append('B' + str(action_b))
        next_state, reward1, reward2, done = couple_move(player_a, player_b, state, action_a, action_b)
        previous_Q = Q1[initial_state, 2, 0].copy()
        if not done:
            v1 = variable()
            x1 = variable()
            x2 = variable()
            x3 = variable()
            x4 = variable()
            x5 = variable()
            constraints1 = []
            prob = [x1, x2, x3, x4, x5]
            for each in prob:
                constraints1.append((each >= 0))
            total = x1 + x2 + x3 + x4 + x5
            constraints1.append((total == 1))
            for b in range(5):
                sum = 0
                for a in range(5):
                    sum += (float(Q1[next_state,a,b]) * prob[a])
                constraints1.append((sum-v1 >= 0))
            lp1 = op(-v1, constraints1)
            lp1.solve()

            v2 = variable()
            y1 = variable()
            y2 = variable()
            y3 = variable()
            y4 = variable()
            y5 = variable()
            constraints2 = []
            prob_y = [y1, y2, y3, y4, y5]
            for each in prob_y:
                constraints2.append((each >= 0))
            total_y = y1+y2+y3+y4+y5
            constraints2.append((total_y == 1))
            for a in range(5):
                sum_y = 0
                for b in range(5):
                    sum_y += (float(Q2[next_state, a, b]) * prob_y[b])
                constraints2.append((sum_y - v2 >= 0))
            lp2 = op(-v2, constraints2)
            lp2.solve()
            list1 = [x1.value[0], x2.value[0], x3.value[0], x4.value[0], x5.value[0]]
            list2 = [y1.value[0], y2.value[0], y3.value[0], y4.value[0], y5.value[0]]
            if next_state == initial_state:
                if time_step > 999500:
                    print("probabilities AGENT A:", "[", x1.value[0],x2.value[0],x3.value[0],x4.value[0],x5.value[0],"]")
                    print("probabilities AGENT B:", list2)
            v1_value = v1.value[0]
            v2_value = v2.value[0]
            Q1[state, action_a, action_b] = (1 - alpha) * Q1[state, action_a, action_b] + alpha * ((1 - gamma) * reward1 + gamma * v1_value)
            Q2[state, action_a, action_b] = (1 - alpha) * Q2[state, action_a, action_b] + alpha * ((1 - gamma) * reward2 + gamma * v2_value)
        elif done:
            Q1[state, action_a, action_b] = (1 - alpha) * Q1[state, action_a, action_b] + alpha * ((1 - gamma) * reward1)
            Q2[state, action_a, action_b] = (1 - alpha) * Q2[state, action_a, action_b] + alpha * ((1 - gamma) * reward2)
        state = next_state
        current_Q = Q1[initial_state, 2, 0].copy()
        diff = abs(current_Q - previous_Q)
        if diff>= 0.0 or time_step> 400000:
            error[time_step] = diff
        time_step += 1
        alpha = max(0.001, alpha * alpha_decay)
        epsilon = max(0.001, decay * epsilon)
    if episode % 10000==0:
        print("EPISODE:", episode,"TIME STEP:", time_step,"EPSILON:", epsilon, "ALPHA", alpha)
    episode += 1

print("Q1[initial_state]:",Q1[initial_state])
print("Q2[initial_state]:",Q2[initial_state])

plot(error)
