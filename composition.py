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
T_states = [(3,3), (3,9), (9,3), (9,9)] # env.possibleStates
dense_goal_rewards = False
goal_reward=2

### Learning base tasks

gamma = 1
maxiter=500
alpha=1
epsilon=0.1

# goals = []
# env = GridWorld(MAP=MAP, goals=goals, T_states=T_states, goal_reward=goal_reward, dense_goal_rewards = dense_goal_rewards)
# MIN,stats1 = Goal_Oriented_Q_learning(env, epsilon=epsilon, alpha=alpha, maxiter=maxiter, p=True)
# save_EQ(MIN,"MIN")

# goals = T_states
# env = GridWorld(MAP=MAP, goals=goals, T_states=T_states, goal_reward=goal_reward, dense_goal_rewards = dense_goal_rewards)
# MAX,stats1 = Goal_Oriented_Q_learning(env, epsilon=epsilon, alpha=alpha, maxiter=maxiter, p=True)
# save_EQ(MAX,"MAX")

# goals = [(3,3), (3,9)]
# env = GridWorld(MAP=MAP, goals=goals, T_states=T_states, goal_reward=goal_reward, dense_goal_rewards = dense_goal_rewards)
# A,stats1 = Goal_Oriented_Q_learning(env, epsilon=epsilon, alpha=alpha, maxiter=maxiter, p=True)
# save_EQ(A,"A")

# goals = [(3,3), (9,3)]
# env = GridWorld(MAP=MAP, goals=goals, T_states=T_states, goal_reward=goal_reward, dense_goal_rewards = dense_goal_rewards)
# B,stats1 = Goal_Oriented_Q_learning(env, epsilon=epsilon, alpha=alpha, maxiter=maxiter, p=True)
# save_EQ(B,"B")

MIN = load_EQ("MIN.npy")
MAX = load_EQ("MAX.npy")
A = load_EQ("A.npy")
B = load_EQ("B.npy")

#####################################################

def get_grid_evfs(env, evf):
    evf_ = np.ones([env.m*env.m,env.n*env.n])
    grid = np.zeros([env.m*env.m,env.n*env.n,4])

    for x in range(env.m):
        for y in range(env.n):
            if (x,y) not in env.walls:
                img = np.zeros([env.m, env.n, 4])
                for (i,j) in env.walls:
                    img[i,j,-1] = 1.0
                grid[x*env.m:x*env.m+env.m,y*env.n:y*env.n+env.n] = img

                img = np.zeros([env.m, env.n])+float("-inf")
                for (i,j) in env.possibleStates:
                    img[i,j] = evf[((i,j))][((x,y))].max()
                    # if img[i,j]>0:
                    #     img[i,j] -= 7
                evf_[x*env.m:x*env.m+env.m,y*env.n:y*env.n+env.n] = img
            else:
                img = np.ones([env.m, env.n, 4])
                grid[x*env.m:x*env.m+env.m,y*env.n:y*env.n+env.n] = img
    evf = evf_[env.m:-env.m,env.n:-env.n]
    grid = grid[env.m:-env.m,env.n:-env.n]

    fig = plt.figure(1, figsize=(20, 20), dpi=60, facecolor='w', edgecolor='k')
    plt.clf()
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.grid(False)
    ax = fig.gca()

    cmap = 'RdBu_r'#'YlOrRd' if False else 'RdYlBu_r'
    ax.imshow(evf, origin="upper", cmap=cmap, extent=[0, env.n, env.m, 0])
    ax.imshow(grid, origin="upper", extent=[0, env.n, env.m, 0])
    plt.show()

    return fig

#####################################################
NEG = lambda EQ: NOT(EQ, EQ_max=MAX, EQ_min=MIN)

Q = SOR(A,B)
fig = env.render( P=EQ_P(Q), V = EQ_V(Q))
fig.savefig("top SOR left.pdf", bbox_inches='tight')
plt.show()
# get_grid_evfs(env, Q)
Q = SAND(A,B)
fig = env.render( P=EQ_P(Q), V = EQ_V(Q))
fig.savefig("top SAND left.pdf", bbox_inches='tight')
plt.show()
# get_grid_evfs(env, Q)
Q = NEG(A)
fig = env.render( P=EQ_P(Q), V = EQ_V(Q))
fig.savefig("NOT top.pdf", bbox_inches='tight')
plt.show()
# get_grid_evfs(env, Q)
Q = NEG(SOR(A,B))
fig = env.render( P=EQ_P(Q), V = EQ_V(Q))
fig.savefig("top SNOR left.pdf", bbox_inches='tight')
plt.show()
# get_grid_evfs(env, Q)
Q = SAND(SOR(A,B),NEG(SAND(A,B)))
fig = env.render( P=EQ_P(Q), V = EQ_V(Q))
fig.savefig("top SXOR left.pdf", bbox_inches='tight')
plt.show()
# get_grid_evfs(env, Q)

