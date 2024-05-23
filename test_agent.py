import wandb
import torch
import numpy as np
import pandas as pd
import json
import datetime, time, os, random, sys
import argparse
from decision_transformer.models.decision_transformer import DecisionTransformer
import DT_GA3C_config as DTconfig

sys.path.append(os.path.abspath(os.path.join('..', 'gym_ca')))
try:
    from gym_collision_avoidance.experiments.src.env_utils import create_env, store_stats
    from gym_collision_avoidance.envs import Config
except:
    print('Could not find gym_collision_avoidance module. Was it installed?')
    sys.exit()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"DEVICE: {DEVICE}")
Config.EVALUATE_MODE = True
Config.SAVE_EPISODE_PLOTS = False
Config.SHOW_EPISODE_PLOTS = False
Config.ANIMATE_EPISODES = False
Config.PLOT_CIRCLES_ALONG_TRAJ = True
Config.MAX_TIME_RATIO = 5


num_agents = DTconfig.num_agents
test_policy = DTconfig.test_policy
other_pols = DTconfig.other_policies

if type(other_pols) == str:
    policies = [other_pols]*num_agents
    policies[0] = test_policy
elif type(other_pols) == list:
    policies = [test_policy]
    policies.extend(other_pols)
    assert(len(policies) == num_agents)
else:
    assert(0), 'Unrecognized policy!'

Config.MAX_NUM_AGENTS_IN_ENVIRONMENT = num_agents
Config.MAX_NUM_AGENTS_TO_SIM = num_agents
Config.MAX_NUM_OTHER_AGENTS_OBSERVED = num_agents - 1
Config.SAVE_EPISODE_PLOTS = True
Config.setup_obs()

env = create_env()

test_cases = pd.read_pickle(
    os.path.dirname(os.path.realpath(__file__)) 
    + f'/decision_transformer/envs/gym_ca/gym_collision_avoidance/envs/test_cases/{num_agents}_agents_500_cases.p'
)

def reset_test(env, case_num, model_path=None, state_dim=None, act_dim=None, checkpt=None):
    import decision_transformer.envs.gym_ca.gym_collision_avoidance.envs.test_cases as tc

    def reset_env(env, agents, case_num, policy,):
        env.unwrapped.plot_policy_name = policy        
        env.set_agents(agents)
        state, _ = env.reset()
        env.unwrapped.test_case_index = case_num
        return state

    if case_num > 500:
        agents = tc.cadrl_test_case_to_agents(test_case=test_cases[case_num-500], policies=policies)
    else:
        agents = tc.cadrl_test_case_to_agents(test_case=test_cases[case_num], policies=policies)

    if model_path is not None:
        [
            agent.policy.initialize_model(state_dim, act_dim, model_path, DEVICE)
            for agent in agents
            if hasattr(agent.policy, "initialize_model")
        ]
    if checkpt is not None:
        [
            agent.policy.initialize_network(checkpt_dir=checkpt[0], checkpt_name=checkpt[1])
            for agent in agents
            if hasattr(agent.policy, "initialize_network")
        ]
    
    return reset_env(env, agents, case_num, policy='DT')


def test_model(env, model_path):
    print(f'Loading DT model: {model_path}')
    state_dim = env.observation_space.shape[1]
    if ('cont' in model_path) == True:
        act_dim = 2
    else:
        act_dim = int(model_path.split('-')[0][-2:])
    DT_agents = [idx for idx in range(len(policies)) if policies[idx] == 'DT']
    env_targets = DTconfig.test_targets
    test_save_dir = os.path.dirname(os.path.realpath(__file__)) + f"/test/{model_path.split('/')[-1].split('.pt')[0]}/{policies[0]}_{policies[1]}_{num_agents}_agents/"
    for tar in env_targets:
        try:
            os.makedirs(test_save_dir + f'/{tar}/', exist_ok=False)
        except:
            print(f'Target {tar} already exists')
            env_targets.remove(tar)
    test_stats = {'Tests': DTconfig.test_episodes, 'Policies': policies}
    test_stats['summary'] = {}
    for tar in env_targets:
        test_stats['summary'][tar] = {'length': 0, 'reward': 0, 'goal': 0}
        test_stats[tar] = {}
        env.set_plot_save_dir(test_save_dir + f'/{tar}/')
        for ep in range(DTconfig.test_episodes):
            print(f'TARGET: {tar} | EP: {ep} / {DTconfig.test_episodes}', end='\r')
            ep_length = 0; ep_reward = 0
            s = reset_test(env, ep, model_path, state_dim, act_dim)
            done = False
            with torch.no_grad():
                for DTA in DT_agents:
                    env.agents[DTA].policy.initialize_history(s[DTA], tar) 
                while not done:
                    state, rew, done, _, stats = env.step([None]) 
                    agent_ts = np.array([(1-int(x)) for x in stats['which_agents_done'].values()])
                    for DTA in DT_agents:
                        if int(agent_ts[DTA]) == 1:
                            env.agents[DTA].policy.model_step(state[DTA], rew[DTA])
                    ep_length += agent_ts
                    ep_reward += rew
                ep_goal = np.array([a.is_at_goal for a in env.agents])
                # Save episodic stats
                test_stats[tar][ep] = {'length': ep_length.tolist(), 'reward': ep_reward.tolist(), 'goal': ep_goal.tolist()}
                # Save total stats
                test_stats['summary'][tar]['length'] += ep_length/DTconfig.test_episodes
                test_stats['summary'][tar]['reward'] += ep_reward/DTconfig.test_episodes
                test_stats['summary'][tar]['goal'] += ep_goal/DTconfig.test_episodes

        env.reset()  # Save last plot

    for tar in env_targets:
        test_stats['summary'][tar]['average'] = {}
        for x in ['length', 'reward', 'goal']:
            test_stats['summary'][tar]['average'][x] = np.mean(test_stats['summary'][tar][x]).tolist()
            test_stats['summary'][tar][x] = test_stats['summary'][tar][x].tolist()

    # Write stats
    print('Saving stats')
    with open(test_save_dir + f'stats{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.json', 'w') as f:
        f.write(json.dumps(test_stats, indent=4))

### MAIN ###
if __name__ == '__main__':

    model_path = os.path.join(os.path.dirname(__file__), DTconfig.model_path)
    seed = DTconfig.seed

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    test_model(env, model_path)
