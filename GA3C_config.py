from decision_transformer.envs.gym_ca.gym_collision_avoidance.envs.policies.GA3C_CADRL.network import Actions, Actions_Plus

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

k = 20
num_eval_episodes = 25
max_iters = 100
num_steps_per_iter = 10000
device = 'cuda'           

### ENV ###
ACTIONS = 'discrete'

if ACTIONS == 'discrete':
    env_name = 'disc_GA3C'
    actions = Actions()   # Actions() or Actions_Plus()
else:
    env_name = 'cont_GA3C'
    
dataset = '4_agent_11_actions/dataset.pkl'
mode = 'normal'
max_ep_len = 1500
scale = 1
state_mean = 0
state_std = 1
gamma = 1.0
print_logs = True

### CHECKPOINT ###
checkpoint = True
resume = False
#load_path = models/mixed_4/20231019-1641.pth


### TEST ###
model_path = 'models/fixed-d4rl-RVO_3-20240226-2306.pt'
env_targets = [0,0.5,1.0]
multi = False
num_agents = 3
eval_episodes = 500
policies = ['external','RVO','RVO']
seed = 0