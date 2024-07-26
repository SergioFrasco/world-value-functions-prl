from GridWorld import *
from library import *
import matplotlib.pyplot as plt

# Environment
goals = [(3,3)] # Task goals
env = GridWorld(goals=goals)

### Load WVF
WVF = load_WVF("WVF.npy")

### Inferring dynamics
states = [(3,3),(3,9),(9,9),(9,3)] # + env.hallwayStates

fig = plt.figure(1, figsize=(20, 20), dpi=60, facecolor='w', edgecolor='k')
params = {'font.size': 40}
plt.rcParams.update(params)
plt.clf()
plt.xticks([])
plt.yticks([])
plt.grid(None)
ax = fig.gca()

for state in states:
    goals = env.get_neightbours(state)
    env.state = state
    inferred_dynamics = WVF_Ta(WVF, state, R=env.get_rewards(), goals=goals, actions = 5)
    env.render(fig=fig, ax=ax, Ta=inferred_dynamics)
fig.savefig("transitions_neightbourhood.png", bbox_inches='tight')
plt.show()
