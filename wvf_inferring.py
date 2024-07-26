from GridWorld import *
from library import *
import matplotlib.pyplot as plt


### Load WVF
WVF = load_WVF("WVF.npy")

### Hallways task
goals = [(2,6),(7,10),(10,6),(6,2)] # Task goals
env = GridWorld(goals=goals, goal_reward=3)
WVF, VF, P = WVF_R_WVF(WVF, env.get_rewards())

fig = env.render(R=env.get_rewards())
fig.savefig("task_2_reward.png", bbox_inches='tight')
plt.show()
fig=env.render(WVF=WVF)
fig.savefig("task_2_wvf.png", bbox_inches='tight')
plt.show()
fig=env.render( P=P, V = VF)
fig.savefig("task_2_vf.png", bbox_inches='tight')
plt.show()

### Bottom task
goals = [(11,1),(11,2),(11,3),(11,4),(11,5),(11,7),(11,8),(11,9),(11,10),(11,11)] # Task goals
env = GridWorld(goals=goals, goal_reward=3, dense_goal_rewards=True)
WVF, VF, P = WVF_R_WVF(WVF, env.get_rewards())

fig = env.render(R=env.get_rewards())
fig.savefig("task_3_reward.png", bbox_inches='tight')
plt.show()
fig=env.render(WVF=WVF)
fig.savefig("task_3_wvf.png", bbox_inches='tight')
plt.show()
fig=env.render( P=P, V = VF)
fig.savefig("task_3_vf.png", bbox_inches='tight')
plt.show()