from decision_transformer.envs.gym_ca.gym_collision_avoidance.envs.policies.GA3C_CADRL.network import Actions, Actions_Plus
from models.models_dict import models
import numpy as np

### MODEL ###
batch_size = 64
model_type = 'dt'
embed_dim = 128
n_layer = 3
n_head = 1
activation_fn = 'relu'
dropout = 0.1
lr = 1e-4
wd = 1e-4
warmup = 10000
wandb = True

k = 5
num_eval_episodes = 50
max_iters = 40
num_steps_per_iter = 2500 # decreased step per iter
device = 'cuda'           

### ENV ###
dataset = '4_agent_11_actions/medium.pkl'
ACTIONS = 'discrete'
# ACTIONS = 'continuous'

if ACTIONS == 'discrete':
    env_name = 'disc_GA3C'
    actions = Actions()   # Actions() or Actions_Plus()
else:
    env_name = 'cont_GA3C'
mode = 'normal'
max_ep_len = 1500
gamma = 1.0
scale = 1
state_mean = 0
state_std = 1
print_logs = True

### EVALUATION ###
num_agents = 4
policies = 'RVO'
env_targets = [0,0.5,1.0]
test_case_fn = "get_testcase_random"
test_case_args = {
    'policy_to_ensure': 'RVO',
    'policies': ['RVO', 'noncoop', 'static', 'random'],
    'policy_distr': [0.9, 0.10, 0, 0],
    'speed_bnds': [0.5, 2.0],
    'radius_bnds': [0.5, 0.5],
    'side_length': [
        {'num_agents': [0,5], 'side_length': [4,6]}, 
        {'num_agents': [5,np.inf], 'side_length': [6,8]},
        ],
    'agents_sensors': ['other_agents_states'],
}

### CHECKPOINT ###
checkpoint = True
resume = False
#load_path = models/4_agent_11_actions/20240409-1337.pth


### TEST ###
model_path = f'models/{models["E-11c-K5-G1"]}'
test_policy = 'DT'
other_policies = 'DT'
test_targets = [-1, -10, 10]
seed = 0
test_episodes = 500