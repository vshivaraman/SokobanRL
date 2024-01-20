import numpy as np
from warehouse import WarehouseAgent
from copy import deepcopy

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
            Q_table[(i[0],i[1],j)] = [0, 0] ## [Q-value, Q updates]

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

def gen_episode(wh_agent, Q_table):
    done = False
    episode_seq = []
    step = 0 
    step_limit = 50

    while(not done and step < step_limit):
        #print('hell')
        agent_pos = deepcopy(wh_agent.agent_position)
        action = choose_action(agent_pos,Q_table)
        #episode_seq.append((agent_pos,action))
        reward,done = wh_agent.step(action)
        step += 1
        #if step == step_limit:
            #    reward = -5
        episode_seq.append((agent_pos,action,reward))

    return episode_seq

def off_policy_MC(wh_agent, Q_table):
    Q_table = init_Q_table(poss_action)
    C_table = init_Q_table(poss_action)

    policy_pi = {}


    for i in range(100000):
        episode_seq = gen_episode(wh_agent, Q_table)
        returns = 0

        rev_episode_seq = reversed(episode_seq)

        W = 1.0

        for index_sa in rev_episode_seq:
            returns = returns*1 + index_sa[2]
            key = (index_sa[0][0],index_sa[0][1],index_sa[1])
            C_table[key][0] += W
            Q_table[key][0] = Q_table[key][0] + W/C_table[key][0]*(index_sa[2] - Q_table[key][0])


            rew = []
            for i in poss_action[(x,y)]:
                rew.append(Q_table[(x,y,i)][0])
            ind_action = np.argmax(np.array(rew))

            policy_pi[(index_sa[0][0],index_sa[0][1])] = ind_action

            if ind_action




            W = W/0.25






