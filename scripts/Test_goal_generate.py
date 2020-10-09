import numpy as np
import yaml
from pathlib import Path

n_goals = 100
range = 0.5

random_goals = np.random.rand(n_goals,2) * range
random_goals[:, 0] += 0.49
random_goals[:, 1] -= range / 2
random_goals = np.round(random_goals, 2)

data = {"goal": random_goals.tolist()}
fname = Path(__file__).parent.parent.absolute() / 'Config' / 'goals.yaml'
with open(fname, "w") as f:
    yaml.dump(data, f, default_flow_style = None)