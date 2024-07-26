import numpy as np
from collections import defaultdict
import itertools


#########################################################################################
def shortest(MAP):
    """
    Use Floyd-Warshall to compute shortest distances between all states
    """
    board = MAP.replace(" ","").split('\n')
    arr = np.array([list(row) for row in board])
    free_spaces = list(map(tuple, np.argwhere(arr != '1')))

    dist = {(x, y) : np.inf for x in free_spaces for y in free_spaces}

    for (u, v) in dist.keys():
        d = abs(u[0] - v[0]) + abs(u[1] - v[1])
        if d == 0:
            dist[(u, v)] = 0
        elif d == 1:
            dist[(u, v)] = 1

    for k in free_spaces:
        for i in free_spaces:
            for j in free_spaces:
                if dist[(i, j)] > dist[(i, k)] + dist[(k, j)]:
                    dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
    
    return dist, free_spaces

def get_vf_optimal(env, dist, free_spaces, gamma = 1):
    V = defaultdict(lambda: 0)
    for state in free_spaces:
        values = []
        for goal in env.T_states:
            d = dist[(state,goal)]
            C = d
            if gamma != 1:
                C = (1-(gamma**C))/(1-gamma)
            v = env._get_reward(goal, 0) + C * env._get_reward(state, 0)
            values.append(v)
        V[state] = np.max(values)
    return V

def get_wvf_optimal(env, dist, free_spaces, rmin_ = -100, gamma = 1):
    wvf = defaultdict(lambda: defaultdict(lambda: 0))
    for state in free_spaces:
        for goal in env.T_states:
            d = dist[(state,goal)]
            C = d
            if gamma != 1:
                C = (1-(gamma**C))/(1-gamma)
            wvf[state][goal] = env._get_reward(goal, 0) + C * env._get_reward(state, 0)
    return wvf

def vf_equal(V1,V2,epsilon=1e-2):    
    for state in V1:
        if abs(V1[state]-V2[state])>epsilon:
            return False
    return True

def wvf_equal(wvf1,wvf2,epsilon=1e-2):    
    for state in wvf1:
        for goal in wvf1[state]:
            if abs(wvf1[state][goal]-wvf2[state][goal])>epsilon:
                return False
    return True


def evaluateVF(env, Q, gamma=1):
    G=0      
    state = env.reset()
    for t in range(100):
        action = Q[state].argmax()
        state, reward, done, _ = env.step(action) 
        G += reward
        if done:
            break
    return G

def evaluateWVF(env, Q, gamma=1):
    G=0      
    state = env.reset()
    goal = goal_policy(Q, state, list(Q[state].keys()), epsilon = 0)
    for t in range(100):
        action = Q[state][goal].argmax()
        state, reward, done, _ = env.step(action) 
        G += reward
        if done:
            break
    return G

#########################################################################################
def to_hash(x, b=False):
    return x

def epsilon_greedy_policy_improvement(env, Q, epsilon = 1):
    """
    Implements policy improvement by acting epsilon-greedily on Q

    Arguments:
    env -- environment with which agent interacts
    Q -- Action function for current policy

    Returns:
    policy_improved -- Improved policy
    """

    def policy_improved(state, goal = None, epsilon = epsilon):
        probs = np.ones(env.action_space.n, dtype=float)*(epsilon/env.action_space.n)
        best_action = np.random.choice(np.flatnonzero(Q[state][goal] == Q[state][goal].max())) #np.argmax(Q[state][goal]) #
        probs[best_action] += 1.0 - epsilon
        return probs
    return policy_improved

def goal_policy(Q, state, goals, epsilon = 0):
    """
    Implements generalised policy improvement
    """
    goal = None
    if goals:
        values = [Q[state][goal].max() for goal in goals]
        values = np.array(values)
        best_goal = np.random.choice(np.flatnonzero(values == values.max()))
        if np.random.random()>epsilon:
            goal = goals[best_goal]
        else:
            goal = goals[np.random.randint(len(goals))]

    return goal

def Q_learning(env, V_optimal=None, gamma=1, epsilon=1, alpha=1, maxiter=100, mean_episodes=100, p=True):
    """
    Implements Q_learning

    Arguments:
    env -- environment with which agent interacts
    gamma -- discount factor
    alpha -- learning rate
    maxiter -- maximum number of episodes

    Returns:
    Q -- New estimate of Q function
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    behaviour_policy =  epsilon_greedy_policy_improvement(env, Q, epsilon = epsilon)
    
    stats = {"R":[], "T":0}
    stats["R"].append(0)
    k=0
    T=0
    t=0    
    state = env.reset()
    state = to_hash(state)

    stop_cond = lambda k: k < maxiter
    if V_optimal:
        stop_cond = lambda k: True if k%mean_episodes != 0 else not vf_equal(V_optimal,Q_V(Q))

    while stop_cond(k):
        probs = behaviour_policy(state, epsilon = epsilon)
        action = np.random.choice(np.arange(len(probs)), p=probs)            
        state_, reward, done, _ = env.step(action)
        state_ = to_hash(state_)
        
        # stats["R"][k] += reward
        
        G = 0 if done else np.max(Q[state_])
        TD_target = reward + gamma*G
        TD_error = TD_target - Q[state][action]
        Q[state][action] = Q[state][action] + alpha*TD_error
        
        state = state_
        t+=1
        if done:            
            stats["R"].append(evaluateVF(env, Q, gamma))
            
            state = env.reset()
            state = to_hash(state)

            stats["T"] += t
            t=0
            k+=1
            if p and k%mean_episodes == 0:
                mean_return = np.mean(stats["R"][-mean_episodes-1:-1])
                print('Episode: ', k, ' | Mean return: ', mean_return,
                      ' | States: ', len(list(Q.keys())))
                    
    return Q, stats
    

def Dyna_Q_learning(env, dyna_steps=10, V_optimal=None, gamma=1, epsilon=1, alpha=1, maxiter=100, mean_episodes=100, p=True):
    """
    Implements Dyna Q_learning

    Arguments:
    env -- environment with which agent interacts
    gamma -- discount factor
    alpha -- learning rate
    maxiter -- maximum number of episodes

    Returns:
    Q -- New estimate of Q function
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    behaviour_policy =  epsilon_greedy_policy_improvement(env, Q, epsilon = epsilon)

    R = defaultdict(lambda: np.zeros(env.action_space.n))
    P = defaultdict(lambda: [None]*env.action_space.n)
    experience = defaultdict(lambda: np.zeros(env.action_space.n))
    
    stats = {"R":[], "T":0}
    stats["R"].append(0)
    k=0
    T=0
    t=0    
    state = env.reset()
    state = to_hash(state)

    stop_cond = lambda k: k < maxiter
    if V_optimal:
        stop_cond = lambda k: True if k%mean_episodes != 0 else not vf_equal(V_optimal,Q_V(Q))

    while stop_cond(k):
        probs = behaviour_policy(state, epsilon = epsilon)
        action = np.random.choice(np.arange(len(probs)), p=probs)            
        state_, reward, done, _ = env.step(action)
        state_ = to_hash(state_)
        
        # stats["R"][k] += reward
        
        G = 0 if done else np.max(Q[state_])
        TD_target = reward + gamma*G
        TD_error = TD_target - Q[state][action]
        Q[state][action] = Q[state][action] + alpha*TD_error

        P[state][action] = state_, done
        R[state][action] = reward
        experience[state][action] = 1
        states = list(experience.keys())
        for _ in range(dyna_steps):
            s = states[np.random.randint(len(states))]
            a = np.random.choice(np.flatnonzero(experience[s] == experience[s].max()))
            (s_, d), r = P[s][a], R[s][a]
            
            G = 0 if d else np.max(Q[s_])
            TD_target = r + gamma*G
            TD_error = TD_target - Q[s][a]
            Q[s][a] = Q[s][a] + alpha*TD_error
        
        state = state_
        t+=1
        if done:            
            stats["R"].append(evaluateVF(env, Q, gamma))

            state = env.reset()
            state = to_hash(state)

            stats["T"] += t
            t=0
            k+=1
            if p and k%mean_episodes == 0:
                mean_return = np.mean(stats["R"][-mean_episodes-1:-1])
                print('Episode: ', k, ' | Mean return: ', mean_return,
                      ' | States: ', len(list(Q.keys())))
                    
    return Q, stats

    
def Goal_Oriented_Q_learning(env, wvf_init=None, wvf_optimal=None, T_states=None, gamma=1, rmin_ = -100, epsilon=1, alpha=1, maxiter=100, mean_episodes=100, p=True):
    """
    Implements Q_learning for WVFs

    Arguments:
    env -- environment with which agent interacts
    T_states -- Absorbing set
    gamma -- discount factor
    alpha -- learning rate
    maxiter -- maximum number of episodes

    Returns:
    Q -- New estimate of Q function
    """
    
    Q = defaultdict(lambda: defaultdict(lambda: np.zeros(env.action_space.n)))
    if wvf_init!=None:
        for s in wvf_init:
            for g,v in wvf_init[s].items():
                Q[s][g] += v
    behaviour_policy =  epsilon_greedy_policy_improvement(env, Q)

    sMem={} # Goals memory
    if T_states:
        for state in T_states:
            state = to_hash(state)
            sMem[state]=0
    goals = list(sMem.keys())

    stats = {"R":[], "T":0}
    stats["R"].append(0)
    k=0
    T=0
    t=0    
    state = env.reset()
    state = to_hash(state)
    goal = goal_policy(Q, state, goals)

    stop_cond = lambda k: k < maxiter
    if wvf_optimal:
        stop_cond = lambda k: True if k%mean_episodes != 0 else not wvf_equal(wvf_optimal,WVF_wvf(Q))

    while stop_cond(k):
        if goal:
            probs = behaviour_policy(state, goal = goal, epsilon = epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)
        else:
            action = env.action_space.sample()
        state_, reward, done, _ = env.step(action)
        state_ = to_hash(state_)

        # stats["R"][k] += reward
        if done:
            sMem[state] = 0
            goals = list(sMem.keys())

        for goal_ in goals:
            if state != goal_ and done:
                reward_ = rmin_
            else:
                reward_ = reward

            G = 0 if done else np.max(Q[state_][goal_])
            TD_target = reward_ + gamma*G
            TD_error = TD_target - Q[state][goal_][action]
            Q[state][goal_][action] = Q[state][goal_][action] + alpha*TD_error

        state = state_
        t+=1
        if done:
            stats["R"].append(evaluateWVF(env, Q, gamma))
            
            state = env.reset()
            state = to_hash(state)
            goal = goal_policy(Q, state, goals, epsilon = 1)

            stats["T"] += t
            t=0
            k+=1
            if p and k%mean_episodes == 0:
                mean_return = np.mean(stats["R"][-mean_episodes-1:-1])                
                print('Episode: ', k, ' | Mean return: ', mean_return,
                      ' | States: ', len(list(Q.keys())), ' | Goals: ', len(goals))

    return Q, stats
        
def Dyna_Goal_Oriented_Q_learning(env, dyna_type=0, dyna_WVF=None, dyna_steps=10, dyna_start=0, wvf_init=None, wvf_optimal=None, T_states=None, gamma=1, rmin_ = -100, epsilon=1, alpha=1, maxiter=100, mean_episodes=100, p=True):
    """
    Implements Goal Oriented Q_learning

    Arguments:
    env -- environment with which agent interacts
    T_states -- Absorbing set
    gamma -- discount factor
    alpha -- learning rate
    maxiter -- maximum number of episodes

    Returns:
    Q -- New estimate of Q function
    """
    
    Q = defaultdict(lambda: defaultdict(lambda: np.zeros(env.action_space.n)))
    if wvf_init!=None:
        for s in wvf_init:
            for g,v in wvf_init[s].items():
                Q[s][g] += v
    behaviour_policy =  epsilon_greedy_policy_improvement(env, Q)

    dyna_WVF = dyna_WVF if dyna_WVF else Q
    R = defaultdict(lambda: np.zeros(env.action_space.n))
    P = defaultdict(lambda: [None]*env.action_space.n)
    experience = defaultdict(lambda: np.zeros(env.action_space.n))

    sMem={} # Goals memory
    if T_states:
        for state in T_states:
            state = to_hash(state)
            sMem[state]=0
    goals = list(sMem.keys())

    stats = {"R":[], "T":0}
    stats["R"].append(0)
    mean_return = 0
    k=0
    T=0
    t=0    
    state = env.reset()
    state = to_hash(state)
    goal = goal_policy(Q, state, goals)

    stop_cond = lambda k: k < maxiter
    if wvf_optimal:
        stop_cond = lambda k: True if k%mean_episodes != 0 else not wvf_equal(wvf_optimal,WVF_wvf(Q))

    while stop_cond(k):
        if goal:
            probs = behaviour_policy(state, goal = goal, epsilon = epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)
        else:
            action = env.action_space.sample()
        state_, reward, done, _ = env.step(action)
        state_ = to_hash(state_)
        
        P[state][action] = state_, done
        R[state][action] = reward
        experience[state][action] = 1
        states = list(experience.keys())

        # stats["R"][k] += reward
        if True: #done:
            sMem[state] = 0
            goals = list(sMem.keys())

        for goal_ in goals:
            if state != goal_ and done:
                reward_ = rmin_
            else:
                reward_ = reward

            G = 0 if done else np.max(Q[state_][goal_])
            TD_target = reward_ + gamma*G
            TD_error = TD_target - Q[state][goal_][action]
            Q[state][goal_][action] = Q[state][goal_][action] + alpha*TD_error

        if k >= dyna_start:
            for _ in range(dyna_steps):
                s = states[np.random.randint(len(states))]
                a = np.random.choice(np.flatnonzero(experience[s] == experience[s].max()))       
                d, r = P[s][a][1], R[s][a]    
                s_ = None        
                if dyna_type == 0:
                    s_ = P[s][a][0]
                elif dyna_type == 1:
                    if a == 4:
                        s_ = state 
                    else:
                        s_ = WVF_T_next(Q, s, a, R=R, goals = env.get_neightbours(s), amax=True)
                        # print(s,a,r,s_,'Episode: ', k, ' | Mean return: ', mean_return,
                        #     ' | States: ', len(list(Q.keys())), ' | Goals: ', len(goals))
                elif dyna_type == 2:
                    if a is not 4: 
                        probs = WVF_T(dyna_WVF, s, a, R=R, goals = env.get_neightbours(s), amax=True)
                        for state,prob in probs.items():
                            if prob==1:
                                s_ = state
                                break
                if s_:
                    for g_ in goals:
                        r_ = rmin_ if (s != g_ and d) else r
                        G = 0 if d else np.max(Q[s_][g_])
                        TD_target = r_ + gamma*G
                        TD_error = TD_target - Q[s][g_][a]
                        Q[s][g_][a] = Q[s][g_][a] + alpha*TD_error

        state = state_
        t+=1
        if done:
            stats["R"].append(evaluateWVF(env, Q, gamma))
            
            state = env.reset()
            state = to_hash(state)
            goal = goal_policy(Q, state, goals, epsilon = 1)

            stats["T"] += t
            t=0
            k+=1
            if p and k%mean_episodes == 0:
                mean_return = np.mean(stats["R"][-mean_episodes-1:-1])                
                print('Episode: ', k, ' | Mean return: ', mean_return,
                      ' | States: ', len(list(Q.keys())), ' | Goals: ', len(goals))

    return Q, stats

#########################################################################################
def WVF_WP(WVF):
    P = defaultdict(lambda: defaultdict(lambda: 0))
    for state in WVF:
        for goal in WVF[state]:
                P[state][goal] = np.argmax(WVF[state][goal])
    return P
def WVF_P(WVF, goal=None):
    P = defaultdict(lambda: 0)
    for state in WVF:
        if goal:
            P[state] = np.argmax(WVF[state][goal])
        else:
            Vs = [WVF[state][goal] for goal in WVF[state].keys()]
            P[state] = np.argmax(np.max(Vs,axis=0))
    return P
def Q_P(Q):
    P = defaultdict(lambda: 0)
    for state in Q:
        P[state] = np.argmax(Q[state])
    return P

def WVF_wvf(WVF):
    V = defaultdict(lambda: defaultdict(lambda: 0))
    for state in WVF:
        for goal in WVF[state]:
                V[state][goal] = np.max(WVF[state][goal])
    return V
def WVF_V(WVF, goal=None):
    V = defaultdict(lambda: 0)
    for state in WVF:
        if goal:
            V[state] = np.max(WVF[state][goal])
        else:
            Vs = [WVF[state][goal] for goal in WVF[state].keys()]
            V[state] = np.max(np.max(Vs,axis=0))
    return V
def wvf_V(wvf, goal=None):
    V = defaultdict(lambda: 0)
    for state in wvf:
        if goal:
            V[state] = wvf[state][goal]
        else:
            Vs = [wvf[state][goal] for goal in wvf[state].keys()]
            V[state] = np.max(Vs)
    return V
def Q_V(Q):
    V = defaultdict(lambda: 0)
    for state in Q:
        V[state] = np.max(Q[state])
    return V

def WVF_Q(WVF, goal=None, actions = 5):
    Q = defaultdict(lambda: np.zeros(actions))
    for state in WVF:
        if goal:
            Q[state] = WVF[state][goal]
        else:
            Vs = [WVF[state][goal] for goal in WVF[state].keys()]
            Q[state] = np.max(Vs,axis=0)
    return Q

def WVF_R_WVF(WVF, R, actions = 5):
    WVF_ = defaultdict(lambda: defaultdict(lambda: np.zeros(actions)))
    VF = defaultdict(lambda: 0)
    P = defaultdict(lambda: 0)
    for s in WVF.keys(WVF):
        Vs = []
        for g in WVF[s].keys():
            wvf = WVF[s][g] + (R[g].max()-WVF[g][g].max())
            WVF_[s][g] = wvf
            Vs.append(wvf.max())
        VF[s] = np.max(Vs)
    return WVF_, VF, P

def WVF_R_approx(WVF, env, gamma=1, actions = 5):
    R = {}
    for state in WVF:
        R[state] = np.zeros(actions)
        for action in range(actions):
            state_ = WVF_T_approx(WVF, state, action, env.get_neightbours(state), next_state=True)
            R[state][action] = WVF[state][state_][action] - WVF[state_][state_].max()
    return R

def WVF_T_approx(WVF_, state, action, goals, type=0, amax=True, next_state=False):
    state_ = state
    if type == 0:
        WVF = np.array([WVF_[state][goal][action] for goal in goals])
        state_ = goals[np.argmax(WVF)]
    else:
        for goal in goals:
            if action==np.argmax(WVF_[state][goal]):
                state_=goal
                break
    if next_state:
        return state_
    
    states = list(WVF_.keys()) 
    T = {state:0 for state in states}
    T[state_] = 1
    return T

def WVF_Ta_approx(WVF, state, goals=None, gamma=1, actions = 5, type=0):
    Ta = defaultdict(lambda: np.zeros(actions))
    for action in range(actions):
        probs = WVF_T_approx(WVF, state, action, goals, type=type, amax=True)
        for s,prob in probs.items():
            Ta[s][action] = prob
    return Ta

def WVF_R(WVF, rmin=-0.1, actions = 5):
    R = {}
    for state in WVF:
        R[state] = np.zeros(actions) + rmin
    return R

def WVF_T_next(WVF_, state, action, R=None, goals=None, rmin_=-100, gamma=1, amax=False):
    T = {}
    R = R if R else WVF_R(WVF_)

    states = list(WVF_.keys()) 
    goals = goals if goals else states
    WVF = np.array([WVF_[state][goal][action] for goal in goals])
    R = np.array([R[state][action] for goal in goals])
    wvf = np.array([[np.max(WVF_[state][goal]) for goal in goals] for state in goals])

    wvfi = np.linalg.pinv(wvf)
    P = (1/gamma)*np.matmul((WVF-R),wvfi)
    next_state = goals[P.argmax()]

    wvf_next = np.array([WVF_[next_state][goal].max() for goal in goals])
    error = np.square(WVF - (R+wvf_next)).mean()
    if error > 1e-5:
        return None

    return next_state 

def WVF_T(WVF_, state, action, R=None, goals=None, rmin_=-100, gamma=1, amax=False):
    T = {}
    R = R if R else WVF_R(WVF_)

    states = list(WVF_.keys()) 
    goals = goals if goals else states
    WVF = np.array([WVF_[state][goal][action] for goal in goals])
    R = np.array([R[state][action] for goal in goals])
    wvf = np.array([[np.max(WVF_[state][goal]) for goal in goals] for state in goals])

    wvfi = np.linalg.pinv(wvf)
    P = (1/gamma)*np.matmul((WVF-R),wvfi)
    if amax:
        next_state = goals[P.argmax()]
        T = {state:0 for state in states}
        T[next_state] = 1.0
    else:
        T = {state:0 for state in states}
        T = {goals[i]:P[i] for i in range(len(goals))}
    return T

def WVF_Ta(WVF, state, R=None, goals=None, gamma=1, actions = 5):
    Ta = defaultdict(lambda: np.zeros(actions))
    for action in range(actions):
        probs = WVF_T(WVF, state, action, R=R, goals=goals, gamma=gamma, amax=True)
        for s,prob in probs.items():
            Ta[s][action] = prob
    return Ta

#########################################################################################

def save_WVF(WVF, path):
    data = [[s,[[g,WVF[s][g]] for g in WVF[s]]] for s in WVF]
    np.save(path,data, allow_pickle=True)

def load_WVF(path, actions = 5):
    data = np.load(path, allow_pickle=True)
    WVF = {s: defaultdict(lambda: np.zeros(actions), {g:v for (g,v) in gv}) for (s,gv) in data}
    WVF = defaultdict(lambda: defaultdict(lambda: np.zeros(actions)), WVF)
    return WVF

#########################################################################################

def WVFMAX(WVF, rmax=2, actions = 5): #Estimating WVF_max
    WVF_max = defaultdict(lambda: defaultdict(lambda: np.zeros(actions)))
    for s in list(WVF.keys()):
        for g in list(WVF[s].keys()):
            c = rmax-max(WVF[g][g])
            if s==g:
                WVF_max[s][g] = WVF[s][g]*0 + rmax
            else:      
                WVF_max[s][g] = WVF[s][g] + c   
    return WVF_max

def WVFMIN(WVF, rmin=-0.1, actions = 5): #Estimating WVF_min
    WVF_min = defaultdict(lambda: defaultdict(lambda: np.zeros(actions)))
    for s in list(WVF.keys()):
        for g in list(WVF[s].keys()):
            c = rmin-max(WVF[g][g])
            if s==g:
                WVF_min[s][g] = WVF[s][g]*0 + rmin
            else:      
                WVF_min[s][g] = WVF[s][g] + c  
    return WVF_min

def NOT(WVF, WVF_max=None, WVF_min=None, actions = 5):
    WVF_max = WVF_max if WVF_max else WVFMAX(WVF)
    WVF_min = WVF_min if WVF_min else WVFMIN(WVF)
    WVF_not = defaultdict(lambda: defaultdict(lambda: np.zeros(actions)))
    for s in list(WVF.keys()):
        for g in list(WVF[s].keys()):
            WVF_not[s][g] = (WVF_max[s][g]+WVF_min[s][g]) - WVF[s][g]    
    return WVF_not

def NEG(WVF, WVF_max=None, WVF_min=None, actions = 5):
    WVF_max = WVF_max if WVF_max else WVF
    WVF_min = WVF_min if WVF_min else defaultdict(lambda: defaultdict(lambda: np.zeros(actions)))
    WVF_not = defaultdict(lambda: defaultdict(lambda: np.zeros(actions)))
    for s in list(WVF_max.keys()):
        for g in list(WVF_max[s].keys()):
            WVF_not[s][g] = WVF_max[s][g] - WVF[s][g]
    return WVF_not

def OR(WVF1, WVF2, actions = 5):
    WVF = defaultdict(lambda: defaultdict(lambda: np.zeros(actions)))
    for s in list(set(list(WVF1.keys())) | set(list(WVF2.keys()))):
        for g in list(set(list(WVF1[s].keys())) | set(list(WVF2[s].keys()))):
            WVF[s][g] = np.max([WVF1[s][g],WVF2[s][g]],axis=0)
    return WVF

def AND(WVF1, WVF2, actions = 5):
    WVF = defaultdict(lambda: defaultdict(lambda: np.zeros(actions)))
    for s in list(set(list(WVF1.keys())) | set(list(WVF2.keys()))):
        for g in list(set(list(WVF1[s].keys())) | set(list(WVF2[s].keys()))):
            WVF[s][g] = np.min([WVF1[s][g],WVF2[s][g]],axis=0)
    return WVF

#########################################################################################
