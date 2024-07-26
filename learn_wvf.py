from GridWorld import *
from library import *
import matplotlib.pyplot as plt

# Environment
goals = [(3,3)] # Task goals
env = GridWorld(goals=goals)
fig = env.render(R=env.get_rewards())
fig.savefig("task_1_reward.pdf", bbox_inches='tight')
plt.show()

### Learning world value function
gamma = 1
maxiter=1000
alpha=1
epsilon=0.1
dyna_steps = 10
dyna_start=0

WVF,stats1 = Goal_Oriented_Q_learning(env, epsilon=epsilon, alpha=alpha, maxiter=maxiter)
save_WVF(WVF,"WVF")

### Plotting value function 
fig=env.render( P=WVF_P(WVF), V = WVF_V(WVF))
fig.savefig("task_vf.pdf", bbox_inches='tight')
plt.show()
plt.plot(stats1["R"])
plt.show()

### Plotting world value function
fig=env.render(WVF=WVF)
fig.savefig("task_wvf.pdf", bbox_inches='tight')
plt.show()
