import numpy as np
from warehouse import WarehouseAgent
from copy import deepcopy
import time
from onp_monte import gen_episode 

alpha = 0.7
gamma = 1
eps = 0.7

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

def sarsa(wh_agent): ## Returns the value function Q
    Q_table = init_Q_table(poss_action)
    c = 0
    for i in range(200000): ## For each episode
        state_1 = deepcopy(wh_agent.agent_position)
        action_1 = choose_action(state_1,Q_table)


        done = False

        while(not done):

            """ The S-A-R-S-A step """
            reward, done = wh_agent.step(action_1)
            state_2 = deepcopy(wh_agent.agent_position)
            action_2 = choose_action(state_2,Q_table)

            key_1 = (state_1[0],state_1[1],action_1)
            key_2 = (state_2[0],state_2[1],action_2)
            #if state_2 == wh_agent.goal_location:
            #    Q_table[key_1][0] = Q_table[key_1][0] + alpha*(reward - Q_table[key_1][0])
            #else:
            #print(key_1,'\n',key_2)
            target = reward + gamma*Q_table[key_2][0]
            predict = Q_table[key_1][0]
            
            if state_2 == wh_agent.goal_location:
                Q_table[key_1][0] += alpha*(reward - predict)

            else:
                Q_table[key_1][0] += alpha*(target - predict)
                Q_table[key_1][1] += 1

            state_1 = deepcopy(state_2)
            action_1 = action_2
        wh_agent.render() 
        c+=1
        print(c)
        wh_agent.reset()

    return Q_table

w = WarehouseAgent()
print(sarsa(w))

#Q_t = {(1, 1, 'down'): [-2.274127261786849, 176115], (1, 1, 'right'): [-8.352639407308121, 115672], (1, 2, 'left'): [-5.483816308022407, 125586], (1, 2, 'down'): [-8.79803021802253, 128738], (2, 1, 'up'): [-5.973237886072746, 166201], (2, 1, 'down'): [-1.0, 0], (2, 1, 'right'): [-8.809841768588022, 167770], (2, 2, 'up'): [-8.711762256810754, 88653], (2, 2, 'left'): [-8.208422908499195, 137113], (2, 2, 'down'): [-4.874089416655702, 112601], (3, 1, 'up'): [-3.1960617904271142, 354813], (3, 1, 'left'): [-1.0, 0], (3, 1, 'down'): [-9.000924557123469, 356283], (3, 2, 'up'): [-7.184447675522165, 42354], (3, 2, 'left'): [-1.0, 0], (3, 2, 'down'): [-9.480298774599945, 42560], (3, 2, 'right'): [-12.137562240492514, 42405], (3, 3, 'left'): [-7.827114422095705, 39602], (3, 3, 'down'): [-10.748886123070756, 23905], (3, 3, 'right'): [-11.42055362567924, 21757], (3, 4, 'left'): [-9.271304528831399, 17068], (3, 4, 'down'): [-13.222009216755621, 11501], (4, 1, 'up'): [-1.0, 0], (4, 1, 'down'): [-9.9466400508043, 138765], (4, 1, 'right'): [-8.15651495429374, 138726], (4, 2, 'up'): [-5.183518120592241, 73761], (4, 2, 'left'): [-6.3253126604520205, 81276], (4, 2, 'down'): [-9.264361977542157, 50007], (4, 2, 'right'): [-13.094753311365105, 48342], (4, 3, 'up'): [-10.40179032576559, 4269], (4, 3, 'left'): [-9.364645724219388, 5339], (4, 3, 'right'): [-13.744177302356558, 3159], (4, 4, 'up'): [-13.904603459361272, 6812], (4, 4, 'left'): [-11.982957295957727, 7848], (5, 1, 'up'): [-3.9669808927871055, 125152], (5, 1, 'right'): [-10.208041806012101, 82776], (5, 2, 'up'): [-6.926574612383873, 63620], (5, 2, 'left'): [-9.153035451616384, 69163]}

#Q_t = {(1, 1, 'down'): [-8.658993696878474, 658785], (1, 1, 'right'): [-9.482507209491336, 476622], (1, 2, 'left'): [-9.740470689430921, 517047], (1, 2, 'down'): [-6.502521084923497, 522496], (2, 1, 'up'): [-10.041373341701371, 618360], (2, 1, 'down'): [-1.0, 0], (2, 1, 'right'): [-7.5844990250675615, 619419], (2, 2, 'up'): [-8.544526724179102, 362922], (2, 2, 'left'): [-5.852804984574097, 515638], (2, 2, 'down'): [-9.356894891511807, 433121], (3, 1, 'up'): [-4.852555059062098, 1144759], (3, 1, 'left'): [-1.0, 0], (3, 1, 'down'): [-7.463720447200382, 1144773], (3, 2, 'up'): [-6.621093931148486, 171548], (3, 2, 'left'): [-1.0, 0], (3, 2, 'down'): [-12.160183644934772, 171855], (3, 2, 'right'): [-10.482104591886529, 172442], (3, 3, 'left'): [-5.041180073083367, 158962], (3, 3, 'down'): [-13.560284152593349, 104784], (3, 3, 'right'): [-12.535934987539509, 97230], (3, 4, 'left'): [-9.03242398914883, 75749], (3, 4, 'down'): [-12.926560681740645, 55730], (4, 1, 'up'): [-1.0, 0], (4, 1, 'down'): [-11.274508092391592, 492669], (4, 1, 'right'): [-12.57652376752752, 492150], (4, 2, 'up'): [-10.822261906162748, 267988], (4, 2, 'left'): [-7.5409103221123805, 294270], (4, 2, 'down'): [-10.007441681568139, 199979], (4, 2, 'right'): [-12.338938979787331, 192452], (4, 3, 'up'): [-9.26071386021643, 20434], (4, 3, 'left'): [-10.97588087850162, 25829], (4, 3, 'right'): [-12.129139771134103, 16983], (4, 4, 'up'): [-12.898176265812308, 34249], (4, 4, 'left'): [-11.996347688529472, 38464], (5, 1, 'up'): [-12.299775073357218, 441928], (5, 1, 'right'): [-10.275529990320628, 320759], (5, 2, 'up'): [-11.836173334604167, 250720], (5, 2, 'left'): [-11.327064484619616, 270018]}
            
count = 0
for i in range(10000):
    #gen_episode(w,Q_t)
    #w.render()                
    #print(i)
#    gen_episode(w,Q_t)
    if gen_episode(w,Q_t)[-1][2] == 1: 
        #w.render()            
        count += 1
        print(count)
    w.reset()

print("Accuracy is:", count/10000*100)
#print(gen_episode(w,Q_t))


