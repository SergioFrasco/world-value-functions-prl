Code for the paper "World Value Functions: Knowledge Representation for Learning and Planning": https://prl-theworkshop.github.io/prl2022-icaps/papers/PRL2022_paper_10.pdf

To learn a world value function, run:
```
python learn_wvf.py
```
<img src="task_1_reward.png"  alt="task reward" width = 25% height = auto > <img src="task_wvf.png"  alt="world value function" width = 25% height = auto > <img src="task_vf.png"  alt="inferred value function" width = 25% height = auto >

To infer transition probabilities using the learned world value function, run:
```
python learn_wvf.py
```
<img src="transitions_neightbourhood.png"  alt="inferred transitions" width = 25% height = auto >

To plan using the inferred transition probabilities, run:
```
python learn_wvf.py
```
<img src="imagined_trajectories.png"  alt="inferred trajectories" width = 25% height = auto >

To reproduce an experiment from the paper, run:
```
python exp.py
```
