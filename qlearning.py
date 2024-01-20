import numpy as np
from warehouse import WarehouseAgent
from copy import deepcopy

eps =  0.75
gamma = 0.9
alpha = 0.3

poss_action = { 
            (1, 1): ('down', 'right'),
            (1, 2): ('left', 'down'),
            (2, 1): ('up', 'down', 'right'),
            (2, 2): ('up', 'left', 'down'),
            (3, 1): ('up', 'left', 'down'),
            (3, 2): ('up', 'left', 'down', 'right'),
            (3, 3): ('left', 'down', 'right'),
            (3, 4): ('left', 'down'),
            (4, 1): ('up', 'down', 'right'),
            (4, 2): ('up', 'left', 'down', 'right'),
            (4, 3): ('up', 'left', 'right'),
            (4, 4): ('up', 'left'),
            (5, 1): ('up', 'right'),
            (5, 2): ('up', 'left')
        }   


def init_Q_table(poss_action):
    Q_table = {}
    for i in poss_action:
        for j in poss_action[i]:
            Q_table[(i[0],i[1],j)] = [np.random.rand(), 0] ## [Q_value, Q updates]

    return Q_table

def choose_action(agent_pos,Q_table): ## Input state, Output action ##
    x = agent_pos[0]
    y = agent_pos[1]

    if eps > np.random.rand():
        rew = []
        for i in poss_action[(x,y)]:
            rew.append(Q_table[(x,y,i)][0])
        ind_action = np.argmax(np.array(rew))

        return poss_action[(x,y)][ind_action]
    else:
        return np.random.choice(poss_action[(x,y)])

def qlearning(wh_agent,Q_table):
    Q_table = init_Q_table(poss_action)

    for i in range(10000):
        state_1 = deepcopy(wh_agent.agent_position)

        done = False

        while(not done):
            action_1 = choose_action(state_1,Q_table)
            reward, done = wh_agent.step(action_1)
            state_2 = deepcopy(wh_agent.agent_position)

            key_1 = (state_1[0],state_1[1],action_1)

            x = state_2[0]
            y = state_2[1]

            rew = []
            for j in poss_action[(x,y)]:
                rew.append(Q_table[(x,y,j)][0])
            Q_s_a_ = np.max(np.array(rew))

            Q_table[key_1][0] = Q_table[key_1][0] + alpha*(reward - gamma*Q_s_a_ - Q_table[key_1][0])
            Q_table[key_1][1] += 1
            state_1 = deepcopy(state_2)
        print(i)
        wh_agent.render()
        wh_agent.reset()

    return Q_table
Q_table = init_Q_table(poss_action)
w = WarehouseAgent()
print(qlearning(w,Q_table))



