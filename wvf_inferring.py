from GridWorld import *
from library import *
import matplotlib.pyplot as plt


### Load WVF
WVF = load_WVF("WVF.npy")

# Inferring WVF from task rewards
goals = [(3,3)] # Task goals
env = GridWorld(goals=goals)
rewards = env.get_rewards()
fig = env.render(R=env.get_rewards())
fig.savefig("task_1_reward.png", bbox_inches='tight')
plt.show()
