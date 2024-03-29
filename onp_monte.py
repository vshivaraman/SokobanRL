import numpy as np
from warehouse import WarehouseAgent
from copy import deepcopy
import time


## Constants ##
eps = 0.75

## Initialize Q_table ##

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

    while(not done):
        agent_pos = deepcopy(wh_agent.agent_position)
        action = choose_action(agent_pos,Q_table)
        #episode_seq.append((agent_pos,action))
        reward,done = wh_agent.step(action)
        episode_seq.append((agent_pos,action,reward))

    return episode_seq


def on_policy_MC(wh_agent,Q_table):
    Q_table = init_Q_table(poss_action)
    start = time.time()
    avg_return = []
    print("this is 100000000000")
    for i in range(10000):
        if i%1000 == 0:
            print(i)
        episode_seq = gen_episode(wh_agent, Q_table)
        returns = 0

        for index_sa in episode_seq:
            #returns += index_sa[2]*0.4
            returns = returns*0.4 + index_sa[2]
            #print(index_sa)
            key = (index_sa[0][0],index_sa[0][1],index_sa[1])
            Q_table[key][0] = (Q_table[key][0]*Q_table[key][1] + returns)/(Q_table[key][1] + 1)
            Q_table[key][1] += 1
        
        wh_agent.render()
        print(i,'return is ',returns)
        avg_return.append(returns)
        wh_agent.reset()
    end = time.time()
    print(end-start)
    x = np.arange(1,len(avg_return),1)
    plt.plot(x,avg_return)
    return Q_table
w = WarehouseAgent()
Q_table = init_Q_table(poss_action)
#print(Q_table)
#print(choose_action(w.agent_position,Q_table))
#print(gen_episode(w,Q_table))
print(on_policy_MC(w,Q_table))


"""
Q_t = {(1, 1, 'down'): [-5.917473716374968, 104723],
(1, 1, 'right'): [-5.943805113463611, 56361],
 (1, 2, 'left'): [-3.524768133810747, 74612],
 (1, 2, 'down'): [-3.4331144560103484, 138586],
 (2, 1, 'up'): [-7.254033675640378, 86472],
(2, 1, 'down'): [-7.11350125765844, 197589],
 (2, 1, 'right'): [-7.176386936366234, 86071],
(2, 2, 'up'): [-5.603532847741192, 56838],
 (2, 2, 'left'): [-5.49922353715248, 130721],
 (2, 2, 'down'): [-5.564265855320356, 57189],
(3, 1, 'up'): [-8.625030632931932, 134659],
(3, 1, 'left'): [-8.365772891555466, 264235],
(3, 1, 'down'): [-8.910129546504058, 179781],
(3, 2, 'up'): [-8.313099612672653, 20138],
(3, 2, 'left'): [-8.34241237420072, 20171],
(3, 2, 'down'): [-8.216108963740677, 54330],
 (3, 2, 'right'): [-8.249436997319174, 20515],
(3, 3, 'left'): [-10.478933850254585, 20635],
(3, 3, 'down'): [-12.652080123266716, 16225],
(3, 3, 'right'): [-11.547507734616829, 11636],
(3, 4, 'left'): [-12.545152975456835, 8923],
 (3, 4, 'down'): [-14.355994058985788, 9426],
 (4, 1, 'up'): [-9.909139429044487, 96680],
(4, 1, 'down'): [-9.718537231968972, 64125],
(4, 1, 'right'): [-9.342664746966516, 111823],
 (4, 2, 'up'): [-9.955031042603322, 37368],
(4, 2, 'left'): [-9.993515639533504, 37629],
(4, 2, 'down'): [-10.099451439319001, 38282],
(4, 2, 'right'): [-9.902863968882029, 101293],
(4, 3, 'up'): [-20.505251730221314, 7369],
 (4, 3, 'left'): [-21.114140508221183, 3345],
(4, 3, 'right'): [-21.0206118206119, 3367],
(4, 4, 'up'): [-15.723759868910898, 6713],
(4, 4, 'left'): [-17.37822368421052, 6080],
(5, 1, 'up'): [-10.361228905236306, 57775],
 (5, 1, 'right'): [-10.414288935416455, 31045],
(5, 2, 'up'): [-10.62752285355767, 44632],
(5, 2, 'left'): [-10.663778092731299, 24695]}
"""


#Q_t = {(1, 1, 'down'): [-2.0001237072348537, 51780], (1, 1, 'right'): [-4.384614559457784, 8076], (1, 2, 'left'): [-3.0515882485936814, 8135], (1, 2, 'down'): [-4.408415688476412, 9546], (2, 1, 'up'): [-3.324406191009392, 51721], (2, 1, 'down'): [-1.0, 0], (2, 1, 'right'): [-3.0517835730303338, 52303], (2, 2, 'up'): [-4.388655813382586, 5605], (2, 2, 'left'): [-2.006174390503149, 34280], (2, 2, 'down'): [-3.900291880778994, 25125], (3, 1, 'up'): [-2.0000000042783057, 539131], (3, 1, 'left'): [-1.0, 0], (3, 1, 'down'): [-3.32286322897834, 537545], (3, 2, 'up'): [-4.860977183655017, 3194], (3, 2, 'left'): [-1.0, 0], (3, 2, 'down'): [-5.067686216567287, 3236], (3, 2, 'right'): [-3.2619588479095643, 3156], (3, 3, 'left'): [-2.0644936158635323, 3120], (3, 3, 'down'): [-4.441507546809749, 463], (3, 3, 'right'): [-5.395484044535246, 339], (3, 4, 'left'): [-3.368322412879122, 268], (3, 4, 'down'): [-6.823447286986763, 87], (4, 1, 'up'): [-1.0, 0], (4, 1, 'down'): [-3.658841990646933, 51219], (4, 1, 'right'): [-4.320659673252237, 50794], (4, 2, 'up'): [-3.594521984238233, 23199], (4, 2, 'left'): [-2.0197671197922555, 29006], (4, 2, 'down'): [-4.392185386411216, 3816], (4, 2, 'right'): [-4.394151706896622, 3971], (4, 3, 'up'): [-3.374590212233378, 44], (4, 3, 'left'): [-5.278995002391582, 53], (4, 3, 'right'): [-6.718236023905843, 13], (4, 4, 'up'): [-6.480163401252678, 16], (4, 4, 'left'): [-5.009162502321183, 84], (5, 1, 'up'): [-2.6117581214145797, 49193], (5, 1, 'right'): [-5.115674360671422, 7831], (5, 2, 'up'): [-3.5579656845695062, 5842], (5, 2, 'left'): [-3.110174653676585, 5805]}


#Q_t = {(1, 1, 'down'): [-1.630746499354142, 12647], (1, 1, 'right'): [-1.5062821438783254, 18693], (1, 2, 'left'): [-1.2016939007238912, 18361], (1, 2, 'down'): [-1.1440179382916595, 115069], (2, 1, 'up'): [-1.65946190160389, 12979], (2, 1, 'down'): [-1.6483059931039916, 4803], (2, 1, 'right'): [-1.64506063731326, 41639], (2, 2, 'up'): [-1.5287910514139418, 14738], (2, 2, 'left'): [-1.5193805105325384, 13807], (2, 2, 'down'): [-1.518154182465401, 138451], (3, 1, 'up'): [-1.660129410351028, 32967], (3, 1, 'left'): [-1.6605587948452913, 3675], (3, 1, 'down'): [-1.6634625999192882, 7355], (3, 2, 'up'): [-1.6153533082329312, 10288], (3, 2, 'left'): [-1.618549824054198, 10812], (3, 2, 'down'): [-1.6381678907793682, 22026], (3, 2, 'right'): [-1.6115693429133462, 120719], (3, 3, 'left'): [-1.649834641520352, 14054], (3, 3, 'down'): [-1.649563186099069, 13826], (3, 3, 'right'): [-1.649476882040835, 135756], (3, 4, 'left'): [-1.6605726728939678, 19588], (3, 4, 'down'): [-1.6604870702184689, 134421], (4, 1, 'up'): [-1.6657380583482089, 24707], (4, 1, 'down'): [-1.6658840785634268, 2826], (4, 1, 'right'): [-1.6658346492169183, 3236], (4, 2, 'up'): [-1.6649348974508524, 11340], (4, 2, 'left'): [-1.665283328158169, 15998], (4, 2, 'down'): [-1.6648965367233672, 135604], (4, 2, 'right'): [-1.6651203021559455, 10871], (4, 3, 'up'): [-1.6658198412181247, 11590], (4, 3, 'left'): [-1.6658096755591778, 115864], (4, 3, 'right'): [-1.6658342811911109, 11944], (4, 4, 'up'): [-1.6643944205220378, 18253], (4, 4, 'left'): [-1.6643658009417717, 128112], (5, 1, 'up'): [-0.013929422697647163, 118784], (5, 1, 'right'): [-1.666417123779769, 16787], (5, 2, 'up'): [-1.6660264730036873, 19646], (5, 2, 'left'): [-1.6660254894766187, 132745]}

#Q_t = {(1, 1, 'down'): [-1.630746499354142, 12647], (1, 1, 'right'): [-1.5062821438783254, 18693], (1, 2, 'left'): [-1.2016939007238912, 18361], (1, 2, 'down'): [-1.1440179382916595, 115069], (2, 1, 'up'): [-1.65946190160389, 12979], (2, 1, 'down'): [-1.6483059931039916, 4803], (2, 1, 'right'): [-1.64506063731326, 41639], (2, 2, 'up'): [-1.5287910514139418, 14738], (2, 2, 'left'): [-1.5193805105325384, 13807], (2, 2, 'down'): [-1.518154182465401, 138451], (3, 1, 'up'): [-1.660129410351028, 32967], (3, 1, 'left'): [-1.6605587948452913, 3675], (3, 1, 'down'): [-1.6634625999192882, 7355], (3, 2, 'up'): [-1.6153533082329312, 10288], (3, 2, 'left'): [-1.618549824054198, 10812], (3, 2, 'down'): [-1.6381678907793682, 22026], (3, 2, 'right'): [-1.6115693429133462, 120719], (3, 3, 'left'): [-1.649834641520352, 14054], (3, 3, 'down'): [-1.649563186099069, 13826], (3, 3, 'right'): [-1.649476882040835, 135756], (3, 4, 'left'): [-1.6605726728939678, 19588], (3, 4, 'down'): [-1.6604870702184689, 134421], (4, 1, 'up'): [-1.6657380583482089, 24707], (4, 1, 'down'): [-1.6658840785634268, 2826], (4, 1, 'right'): [-1.6658346492169183, 3236], (4, 2, 'up'): [-1.6649348974508524, 11340], (4, 2, 'left'): [-1.665283328158169, 15998], (4, 2, 'down'): [-1.6648965367233672, 135604], (4, 2, 'right'): [-1.6651203021559455, 10871], (4, 3, 'up'): [-1.6658198412181247, 11590], (4, 3, 'left'): [-1.6658096755591778, 115864], (4, 3, 'right'): [-1.6658342811911109, 11944], (4, 4, 'up'): [-1.6643944205220378, 18253], (4, 4, 'left'): [-1.6643658009417717, 128112], (5, 1, 'up'): [-0.013929422697647163, 118784], (5, 1, 'right'): [-1.666417123779769, 16787], (5, 2, 'up'): [-1.6660264730036873, 19646], (5, 2, 'left'): [-1.6660254894766187, 132745]}

#Q_t = {(1, 1, 'down'): [-996.155973923195, 16413], (1, 1, 'right'): [-39.33813042912958, 77130], (1, 2, 'left'): [-123.85038236041407, 31384], (1, 2, 'down'): [-7.854133763531261, 211986], (2, 1, 'up'): [-248.71250180984558, 62159], (2, 1, 'down'): [-36.35860263837528, 57308], (2, 1, 'right'): [-11.910002424793527, 569122], (2, 2, 'up'): [-36.5391343729668, 66241], (2, 2, 'left'): [-9.073707131181166, 661320], (2, 2, 'down'): [-11.661650954798777, 66192], (3, 1, 'up'): [-156.19344141489165, 10856], (3, 1, 'left'): [-29.085234963302234, 10491], (3, 1, 'down'): [-15.30750451369752, 101912], (3, 2, 'up'): [-34.180213523131656, 12645], (3, 2, 'left'): [-20.69438860308688, 12635], (3, 2, 'down'): [-21.27399607072697, 12725], (3, 2, 'right'): [-19.324824421536103, 164314], (3, 3, 'left'): [-20.57813851658036, 18727], (3, 3, 'down'): [-20.453751279151284, 18567], (3, 3, 'right'): [-20.22775518088987, 186358], (3, 4, 'left'): [-22.29038915094314, 27136], (3, 4, 'down'): [-20.819385183400072, 186885], (4, 1, 'up'): [-20.11064565090492, 42825], (4, 1, 'down'): [-17.05455562548112, 42903], (4, 1, 'right'): [-16.67328882864068, 430816], (4, 2, 'up'): [-24.714913117546555, 117400], (4, 2, 'left'): [-24.330495768118688, 1528517], (4, 2, 'down'): [-24.507311855341925, 117795], (4, 2, 'right'): [-24.48498810058156, 118073], (4, 3, 'up'): [-23.75392626186253, 23814], (4, 3, 'left'): [-23.626866128086053, 235434], (4, 3, 'right'): [-23.840841678286314, 23667], (4, 4, 'up'): [-23.153323934497664, 27663], (4, 4, 'left'): [-21.317861653789333, 182889], (5, 1, 'up'): [-22.137409381310277, 120836], (5, 1, 'right'): [-30.3108308730075, 202209], (5, 2, 'up'): [-28.86186342882944, 39862], (5, 2, 'left'): [-28.533642224300657, 280142]}

#Q_t = {(1, 1, 'down'): [-946.2860916864457, 10296], (1, 1, 'right'): [-36.32976299838872, 43544], (1, 2, 'left'): [-50.66816555948799, 43054], (1, 2, 'down'): [-3.2457645136994953, 294429], (2, 1, 'up'): [-845.7168180975079, 10786], (2, 1, 'down'): [-155.32942442173825, 7436], (2, 1, 'right'): [-30.605463013222092, 69998], (2, 2, 'up'): [-32.32084660904108, 43940], (2, 2, 'left'): [-10.862349830692274, 43118], (2, 2, 'down'): [-8.990260541987006, 430065], (3, 1, 'up'): [-56.73195509478324, 32602], (3, 1, 'left'): [-30.278864787147572, 31994], (3, 1, 'down'): [-25.848995089731375, 321571], (3, 2, 'up'): [-18.516782830813128, 155814], (3, 2, 'left'): [-18.034460002703042, 155473], (3, 2, 'down'): [-17.967204681372472, 155510], (3, 2, 'right'): [-17.919235705741283, 2020269], (3, 3, 'left'): [-18.21949763184256, 1853979], (3, 3, 'down'): [-18.556922135702422, 186337], (3, 3, 'right'): [-18.29957020825243, 186602], (3, 4, 'left'): [-19.06395331437948, 26732], (3, 4, 'down'): [-18.947990404171506, 186331], (4, 1, 'up'): [-31.623799564999224, 191264], (4, 1, 'down'): [-31.30342341964985, 1911130], (4, 1, 'right'): [-31.38585568277011, 192063], (4, 2, 'up'): [-29.7847368757917, 203117], (4, 2, 'left'): [-34.96719771995802, 243855], (4, 2, 'down'): [-29.257792247826412, 2595400], (4, 2, 'right'): [-29.78816384888719, 203529], (4, 3, 'up'): [-26.12946250299485, 25042], (4, 3, 'left'): [-25.324932886436855, 251812], (4, 3, 'right'): [-26.104817075541337, 25721], (4, 4, 'up'): [-20.416991043423135, 26461], (4, 4, 'left'): [-20.187437968434946, 185591], (5, 1, 'up'): [-31.436703942063613, 1971810], (5, 1, 'right'): [-31.58073180122453, 301284], (5, 2, 'up'): [-29.887238669359633, 2534720], (5, 2, 'left'): [-29.984570841298567, 361964]}
#Q_t = {(1, 1, 'down'): [-235.2900027243306, 47723], (1, 1, 'right'): [-249.1045289509567, 5233], (1, 2, 'left'): [-73.26979351443204, 28186], (1, 2, 'down'): [-1.4975651260767402, 245639], (2, 1, 'up'): [-421.717561566042, 24770], (2, 1, 'down'): [-5.606237790420982, 266697], (2, 1, 'right'): [-41.62412600842566, 20824], (2, 2, 'up'): [-54.85660194700995, 18593], (2, 2, 'left'): [-1.7407918726534517, 235846], (2, 2, 'down'): [-4.9921663019727145, 18280], (3, 1, 'up'): [-36.337009957518816, 28722], (3, 1, 'left'): [-10.523602874481677, 28666], (3, 1, 'down'): [-8.58262973660186, 371946], (3, 2, 'up'): [-28.458375959081238, 6256], (3, 2, 'left'): [-18.88347656883421, 6167], (3, 2, 'down'): [-17.8959477422037, 105324], (3, 2, 'right'): [-18.701207535018515, 6211], (3, 3, 'left'): [-26.448583523282473, 6142], (3, 3, 'down'): [-22.961916214553696, 71516], (3, 3, 'right'): [-38.33661120644599, 13403], (3, 4, 'left'): [-40.101766889383875, 13470], (3, 4, 'down'): [-49.685868911605475, 1663], (4, 1, 'up'): [-19.190256955960827, 127804], (4, 1, 'down'): [-18.797945981163274, 128918], (4, 1, 'right'): [-18.76091416818936, 1662495], (4, 2, 'up'): [-20.015158334666936, 99536], (4, 2, 'left'): [-21.5409260632835, 1543415], (4, 2, 'down'): [-20.187160274440313, 101295], (4, 2, 'right'): [-10.012944459006953, 251598], (4, 3, 'up'): [-103.48029197080325, 137], (4, 3, 'left'): [-93.0528683914513, 1778], (4, 3, 'right'): [-133.12660550458673, 218], (4, 4, 'up'): [-59.49109826589581, 1730], (4, 4, 'left'): [-62.794701986755115, 151], (5, 1, 'up'): [-21.062195923735665, 16731], (5, 1, 'right'): [-20.678377238086387, 151692], (5, 2, 'up'): [-20.069193655657457, 213482], (5, 2, 'left'): [-25.272395899254192, 39505]}

#Q_t = {(1, 1, 'down'): [-3.029735812460125, 9465], (1, 1, 'right'): [-2.0880938519387238, 16260], (1, 2, 'left'): [-1.508805504345471, 17490], (1, 2, 'down'): [-1.2778909275895636, 109524], (2, 1, 'up'): [-3.1974238368899064, 8235], (2, 1, 'down'): [-2.5148637458186327, 12512], (2, 1, 'right'): [-2.7990803768147945, 1952], (2, 2, 'up'): [-2.0680743379912103, 10754], (2, 2, 'left'): [-1.9663027101335544, 9878], (2, 2, 'down'): [-1.9489899870140068, 97897], (3, 1, 'up'): [-2.933702551966019, 3356], (3, 1, 'left'): [-2.8529435167796193, 2784], (3, 1, 'down'): [-2.826134962579229, 26896], (3, 2, 'up'): [-2.458530466436097, 7053], (3, 2, 'left'): [-2.5096635915454537, 12913], (3, 2, 'down'): [-2.438451722763792, 86273], (3, 2, 'right'): [-2.4416793151141283, 7002], (3, 3, 'left'): [-2.7611651716814865, 7054], (3, 3, 'down'): [-2.810586242964681, 745], (3, 3, 'right'): [-2.7935505578667796, 765], (3, 4, 'left'): [-2.961751582385907, 695], (3, 4, 'down'): [-2.9874995755908986, 95], (4, 1, 'up'): [-3.105367824266801, 4827], (4, 1, 'down'): [-3.056922604651108, 34104], (4, 1, 'right'): [-3.179414713070422, 16772], (4, 2, 'up'): [-2.891250899685192, 8293], (4, 2, 'left'): [-3.0648898936181284, 14490], (4, 2, 'down'): [-2.9451027419800564, 9564], (4, 2, 'right'): [-2.8546948905779383, 100030], (4, 3, 'up'): [-3.2356043532822647, 132], (4, 3, 'left'): [-3.291640437784442, 45], (4, 3, 'right'): [-3.257649746983252, 16], (4, 4, 'up'): [-3.216199182950841, 25], (4, 4, 'left'): [-3.0901095275492567, 86], (5, 1, 'up'): [-3.2497834012938407, 14331], (5, 1, 'right'): [-3.239085816359523, 81631], (5, 2, 'up'): [-3.1726860005122868, 29337], (5, 2, 'left'): [-3.2804431865331436, 61858]}


count = 0
c = 0
for i in range(10000):
    #gen_episode(w,Q_t)
    #w.render()
    if gen_episode(w,Q_t)[-1][2] == 1:
        #w.render()
        count += 1
    w.reset()

print(count/10000*100)
print(gen_episode(w,Q_t))

