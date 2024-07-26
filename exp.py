from GridWorld import *
from library import *
import matplotlib.pyplot as plt

MAP =   "1 1 1 1 1 1 1 1 1 1 1 1 1\n" \
        "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
        "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
        "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
        "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
        "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
        "1 1 0 1 1 1 1 0 0 0 0 0 1\n" \
        "1 0 0 0 0 0 1 1 1 1 0 1 1\n" \
        "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
        "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
        "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
        "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
        "1 1 1 1 1 1 1 1 1 1 1 1 1"

env = GridWorld(MAP=MAP)
T_states = env.possibleStates
goal_reward=10
goals = [(3,3)]
env = GridWorld(MAP=MAP, goals=goals, T_states=T_states, goal_reward=goal_reward)

### Learning
WVF = load_WVF("WVF.npy")

gamma = 1
maxiter=500
alpha=1
epsilon=0.1
dyna_steps = 10
dyna_start=500

runs = 25
algo = "Ql_WVF"

for run in range(runs):
    print(run)
    if algo == "Ql":
        _,stats1 = Q_learning(env, epsilon=epsilon, alpha=alpha, maxiter=maxiter, p=False)
    elif algo == "Dyna_Ql":
        _,stats1 = Dyna_Q_learning(env, dyna_steps=dyna_steps, epsilon=epsilon, alpha=alpha, maxiter=maxiter, p=False)
    elif algo == "Ql_WVF":
        _,stats1 = Goal_Oriented_Q_learning(env, epsilon=epsilon, alpha=alpha, maxiter=maxiter, p=True)
    elif algo == "Dyna_Ql_WVF":
        _,stats1 = Dyna_Goal_Oriented_Q_learning(env, dyna_steps=dyna_steps, dyna_type=1
                                , epsilon=epsilon, alpha=alpha, maxiter=maxiter, p=True)
    elif algo == "Dyna_Ql_WVF_pretrained":
        _,stats1 = Dyna_Goal_Oriented_Q_learning(env, dyna_steps=dyna_steps, dyna_WVF=WVF, dyna_type=2
                                , epsilon=epsilon, alpha=alpha, maxiter=maxiter, p=True)

    data_path = "data/algo_{}.run_{}".format(algo,run)
    np.save(data_path, stats1["R"])
