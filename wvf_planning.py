from GridWorld import *
from library import *
import matplotlib.pyplot as plt

# Environment
goals = [(3,3)] # Task goals
env = GridWorld(goals=goals)

### Load WVF
WVF = load_WVF("WVF.npy")

### Planning using inferred dynamics
fig = plt.figure(1, figsize=(20, 20), dpi=60, facecolor='w', edgecolor='k')
params = {'font.size': 40}
plt.rcParams.update(params)
plt.clf()
plt.xticks([])
plt.yticks([])
plt.grid(None)
ax = fig.gca()

grid = env._gridmap_to_img()  
ax.imshow(grid, origin="upper", extent=[0, env.n, env.m, 0])

states = [(1,7),(3,11),(10,1),(11,5),(11,11)]
P = WVF_P(WVF, goal=(3,3))
V = WVF_V(WVF, goal=(3,3))
vmin, vmax = min(list(V.values())), max(list(V.values()))
for (x,y) in states:
    state = x,y
    for _ in range(20):
        action = P[state]
        env._draw_action(ax, state[1], state[0], action, color=[1-(V[state]-vmin)/(vmax-vmin)]*3)
        if action == DONE:
            break

        # Next state
        probs = WVF_T(WVF, state, action, goals=env.get_neightbours(state), amax=True)
        for s,prob in probs.items():
            if prob==1:
                state = s
                break
fig.savefig("imagined_trajectories.pdf", bbox_inches='tight')
plt.show()