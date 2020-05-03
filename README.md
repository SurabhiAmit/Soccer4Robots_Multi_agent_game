Project description:

A Soccer game "RoboCup" in which intelligent agents trained using deep reinforcement learning techniques of regular Q-learning, correlated Q-learning, foe-Q learning and friend-Q learning are made to play and their performances observed and compared. The research papers regarding the algorithm implementations are also attached.

Code description:

soccer.py: The environment for the grid soccer game is coded in soccer.py. Other files use this code as a testbed for their Q-learning algorithms.

ceQ.py trains 2 players to play the grid soccer game using correlated Q-learning algorithm. The agent is trained for 1 million timesteps with the parameters: learning rate initialised to 1 and decayed to 0.001 by the end of the training phase. Off-policy training is used, with epsilon value set to 1.0 throughout the run. The discount factor is 0.9. The plot to be replicated gets generated.

foeQ.py trains 2 players to play the grid soccer game using foe Q-learning algorithm. The agent is trained for 1 million timesteps with the parameters, learning rate initialised to 1 and decayed to 0.001 by the end of the training phase. Off-policy training is used, with epsilon value set to 1.0 throughout the run. The discount factor is 0.9. The plot to be replicated gets generated.

friendQ.py trains 2 players to play the grid soccer game using friend Q-learning algorithm. The agent is trained for 1 million timesteps with the parameters, learning rate is 0.3 throughout the run. On-policy epsilon-greedy approach is used, where epsilon is initialised to 1.0 and decayed to 0.001 by the end of the training period. The discount factor is 0.9. The plot to be replicated gets generated.

RegularQ.py trains 2 players to play the grid soccer game using Regular Q-learning algorithm. The agent is trained for 1 million timesteps with the parameters, learning rate initialised to 0.6 and decayed by a factor of 0.999996. On-policy epsilon-greedy approach is used, where epsilon is initialised to 0.9 and decayed by a factor of 0.999995. The discount factor is 0.9. The plot to be replicated gets generated.

Environment used to develop code: 
Python 3.5, Pycharm IDE, Windows10

Imported libraries:
numpy, random, matplotlib(for visualization), cvxopt(for linear programming), copy

How to run the source code?

To run CE-Q learner implementation:
1. Open ceQ.py in IDE like PyCharm and click 'Run'.
2. Or, on command prompt, go to the location where ceQ.py is stored and type "ceQ.py" and hit Enter key.
3. The graph showing the Q-value difference for player A choosing to move South in the initial state, when player B chooses to Stick, gets generated.
4. Towards the end of the training phase, the probabilities of the agents choosing each posssible combination of action in the initial state, are printed.

To run Foe Q-learner implementation:
1. Open foeQ.py in IDE like PyCharm and click 'Run'.
2. Or, on command prompt, go to the location where foeQ.py is stored and type "foeQ.py" and hit Enter key.
3. The graph showing the Q-value difference for player A choosing to move South in the initial state, when player B chooses to Stick, gets generated.
4. Towards the end of the training phase, the probabilities of the agents choosing each posssible action in the initial state, are printed.

To run Friend Q-learner implementation:
1. Open friendQ.py in IDE like PyCharm and click 'Run'.
2. Or, on command prompt, go to the location where friendQ.py is stored and type "friendQ.py" and hit Enter key.
3. The graph showing the Q-value difference for player A choosing to move South in the initial state, when player B chooses to Stick, gets generated.
4. On closing the graph, the graph showing the non-zero error values(till around 350K iterations) gets generated.
5. After the training phase, the Q1[initial_state] for player A and Q2[initial_state] for player B will be printed on the console.

To run Regular Q-learner implementation:
1. Open RegularQ.py in IDE like PyCharm and click 'Run'.
2. Or, on command prompt, go to the location where RegularQ.py is stored and type "RegularQ.py" and hit Enter key.
3. The graph showing the Q-value difference for player A choosing to move South in the initial state  gets generated.
4. The rows in Q-tables of both the players, corresponding to the initial state, are printed in the console.