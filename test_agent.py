import wandb
import torch
import numpy as np
import pandas as pd
import datetime, time, os, random
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.evaluation.evaluate_episodes import test_episode_rtg_grid
import configparser
from decision_transformer.envs.gym_ca_data_gen.gym_collision_avoidance.experiments.src.env_utils import create_env, store_stats
from decision_transformer.envs.gym_ca_data_gen.gym_collision_avoidance.envs import Config

DTConf = configparser.ConfigParser()
DTConf.read('experiment.config')
rewConf = configparser.ConfigParser()
rewConf.read('decision_transformer/envs/gym_ca_data_gen/reward.config')

Config.EVALUATE_MODE = True
Config.SAVE_EPISODE_PLOTS = True
Config.SHOW_EPISODE_PLOTS = True
Config.ANIMATE_EPISODES = False
Config.DT = 0.1
Config.PLOT_CIRCLES_ALONG_TRAJ = True
Config.MAX_TIME_RATIO = 3

rewardtype = DTConf['test']['reward']
Config.REWARD_AT_GOAL = rewConf.getfloat(rewardtype, 'reach_goal')
Config.REWARD_COLLISION_WITH_AGENT = rewConf.getfloat(rewardtype, 'collision_agent')
Config.REWARD_COLLISION_WITH_WALL = -rewConf.getfloat(rewardtype, 'collision_wall')
Config.REWARD_GETTING_CLOSE   = rewConf.getfloat(rewardtype, 'close_reward')
Config.GETTING_CLOSE_RANGE = rewConf.getfloat(rewardtype, 'close_range')
Config.REWARD_TIME_STEP   = rewConf.getfloat(rewardtype, 'timestep')
Config.REACHER = rewConf.getboolean(rewardtype, 'reacher')

policies = [str(s) for s in DTConf['test']['policies'].split(',')]
num_agents = DTConf.getint('test', 'num_agents')
test_cases = pd.read_pickle(
    os.path.dirname(os.path.realpath(__file__)) 
    + f'/decision_transformer/envs/gym_ca_data_gen/gym_collision_avoidance/envs/test_cases/{num_agents}_agents_500_cases.p'
)

def reset_test(env, case_num, envs=None):
    import decision_transformer.envs.gym_ca_data_gen.gym_collision_avoidance.envs.test_cases as tc

    def reset_env(env, agents, case_num, policy,):
        env.unwrapped.plot_policy_name = policy        
        env.set_agents(agents)
        init_obs = env.reset()
        env.unwrapped.test_case_index = case_num
        return init_obs, agents
    
    def reset_envs(envs, agents, case_num, policy,):
        for env in envs:
            env.unwrapped.plot_policy_name = policy
            env.set_agents(agents)
            init_obs = env.reset()
            env.unwrapped.test_case_index = case_num
            agents.append(agents.pop(0))    # rotate agents 1 to left
        return init_obs, agents

    if case_num > 500:
        agents = tc.cadrl_test_case_to_agents(test_case=test_cases[case_num-501], policies=policies)
    else:
        agents = tc.cadrl_test_case_to_agents(test_case=test_cases[case_num-1], policies=policies)
    
    if envs is None:
        _, _ = reset_env(env, agents, case_num, policy='DT')
    else:
        _, _ = reset_envs(envs, agents, case_num, policy='DT')

# Model from models dir
def load_model(model_path):
    model = DecisionTransformer(
                state_dim=DTConf.getint('model','state_dim'),
                act_dim=DTConf.getint('model','action_dim'),
                res=DTConf.getint('env', 'res'),
                max_length=DTConf.getint('model','k'),
                max_ep_len=DTConf.getint('env', 'max_ep_len'),
                hidden_size=DTConf.getint('model','embed_dim'),
                n_layer=DTConf.getint('model','n_layer'),
                n_head=DTConf.getint('model','n_head'),
                n_inner=4*DTConf.getint('model','embed_dim'),
                activation_function=DTConf['model']['activation_fn'],
                n_positions=1024,
                resid_pdrop=DTConf.getfloat('model','dropout'),
                attn_pdrop=DTConf.getfloat('model','dropout'),
            )
    model.load_state_dict(torch.load(model_path))
    return model

### EVALUATION of loaded model ###
def test_single(model_path):
    model = load_model(model_path)
    env = create_env()
    env_targets = [float(s) for s in DTConf['test']['env_targets'].split(',')]
    test_save_dir = os.path.dirname(os.path.realpath(__file__)) + f"/test/tmp/{model_path.split('-')[1]}/{policies[1]}_{num_agents}_agents/{model_path.split('.pt')[0][-13:]}/"
    print(test_save_dir)
    os.makedirs(test_save_dir, exist_ok=True)
    for tar in env_targets:
        os.makedirs(test_save_dir + f'/{tar}/', exist_ok=True)

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            print()
            env.set_plot_save_dir(test_save_dir + f'/{target_rew}/')
            reset_test(env, 1)
            df = pd.DataFrame()
            for x in range(DTConf.getint('test','eval_episodes')):
                print(f"RTG: {target_rew:.1f} | Eval episode: {x+1} / {DTConf.getint('test','eval_episodes')}", end='\r')
                with torch.no_grad():
                    if DTConf['model']['model_type'] == 'dt':
                        ret, length, stats = test_episode_rtg_grid(
                            env,
                            act_dim=DTConf.getint('model', 'action_dim'),
                            model=model,
                            max_ep_len=DTConf.getint('env', 'max_ep_len'),
                            scale=DTConf.getfloat('env', 'scale'),
                            target_return=target_rew/DTConf.getfloat('env', 'scale'),
                            mode=DTConf['env']['mode'],
                            state_mean=DTConf.getfloat('env', 'state_mean'),
                            state_std=DTConf.getfloat('env', 'state_std'),
                            device=DTConf['model']['device'],
                            RES=DTConf.getint('env', 'res')
                        )
                        df = store_stats(
                            df,
                            {"test_episode": x+1, "target_return": target_rew},
                            stats,
                        )
                        reset_test(env, x+2)
                returns.append(ret)
                lengths.append(length)
            log_filename = test_save_dir + f"/stats.p"
            if os.path.isfile(log_filename):
                dfs = []
                oldDF = pd.read_pickle(log_filename)
                dfs.append(oldDF)
                dfs.append(df)
                newDF = pd.concat(dfs, ignore_index=True)
                newDF.to_pickle(log_filename)
            else:
                df.to_pickle(log_filename)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn
    logs = dict()
    diagnostics = dict()
    eval_start = time.time()

    model.eval()
    eval_fns=[eval_episodes(tar) for tar in env_targets]
    for eval_fn in eval_fns:
        outputs = eval_fn(model)
        for k, v in outputs.items():
            logs[f'evaluation/{k}'] = v

    logs['time/total'] = time.time() - eval_start
    logs['time/evaluation'] = time.time() - eval_start

    for k in diagnostics:
        logs[k] = diagnostics[k]

    if DTConf.getboolean('env','print_logs'):
        print()
        print('=' * 80)
        for k, v in logs.items():
            print(f'{k}: {v}')


def test_multi(model_path):
    # Load copy of model for each agent to get actions from
    num_agents = 4
    models = []
    envs = []
    for i in range(num_agents):
        models.append(load_model(model_path))
        envs.append(create_env())
    
    # model_evaluation(env, models, model_path)

### MAIN ###
if __name__ == '__main__':
    model_path = DTConf['test']['model_path']
    seed = DTConf.getint('test', 'seed')
    multi = DTConf.getboolean('test', 'multi')

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if multi:
        test_multi(model_path)
    else:
        test_single(model_path)