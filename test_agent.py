import wandb
import torch
import numpy as np
import pandas as pd
import datetime, time, os, random
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg_grid
import configparser
from decision_transformer.envs.gym_ca_data_gen.gym_collision_avoidance.experiments.src.env_utils import create_env, store_stats
from decision_transformer.envs.gym_ca_data_gen.gym_collision_avoidance.envs import Config

DTConf = configparser.ConfigParser()
DTConf.read('experiment.config')

Config.EVALUATE_MODE = True
Config.SAVE_EPISODE_PLOTS = True
Config.SHOW_EPISODE_PLOTS = True
Config.DT = 0.1
Config.PLOT_CIRCLES_ALONG_TRAJ = True

policies = [str(s) for s in DTConf['test']['policies'].split(',')]
num_agents = DTConf.getint('test', 'num_agents')
test_cases = pd.read_pickle(
    os.path.dirname(os.path.realpath(__file__)) 
    + f'/decision_transformer/envs/gym_ca_data_gen/gym_collision_avoidance/envs/test_cases/{num_agents}_agents_500_cases.p'
)

def reset_test(case_num):
    from decision_transformer.envs.gym_ca_data_gen.gym_collision_avoidance.experiments.src.create_dataset import reset_env
    import decision_transformer.envs.gym_ca_data_gen.gym_collision_avoidance.envs.test_cases as tc

    def reset_env(env, agents, case_num, policy,):
        env.unwrapped.plot_policy_name = policy        
        env.set_agents(agents)
        init_obs = env.reset()
        env.unwrapped.test_case_index = case_num
        return init_obs, agents

    if case_num > 500:
        agents = tc.cadrl_test_case_to_agents(test_case=test_cases[case_num-501], policies=policies)
    else:
        agents = tc.cadrl_test_case_to_agents(test_case=test_cases[case_num-1], policies=policies)

    _, _ = reset_env(env, agents, case_num, policy='DT')

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
def model_evaluation(env, model_path):
    model = load_model(model_path)
    env_targets = [float(s) for s in DTConf['test']['env_targets'].split(',')]
    eval_save_dir = os.path.dirname(os.path.realpath(__file__)) + f"/evaluation/{model_path.split('-')[1]}/{policies[1]}_{num_agents}_agents/{model_path.split('.pt')[0][-13:]}/"
    print(eval_save_dir)
    os.makedirs(eval_save_dir, exist_ok=True)
    for tar in env_targets:
        os.makedirs(eval_save_dir + f'/{tar}/', exist_ok=True)

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            print()
            env.set_plot_save_dir(eval_save_dir + f'/{target_rew}/')
            reset_test(1)
            df = pd.DataFrame()
            for x in range(DTConf.getint('test','eval_episodes')):
                print(f"RTG: {target_rew:.1f} | Eval episode: {x+1} / {DTConf.getint('test','eval_episodes')}", end='\r')
                with torch.no_grad():
                    if DTConf['model']['model_type'] == 'dt':
                        ret, length, stats = evaluate_episode_rtg_grid(
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
                            {"eval_episode": x+1, "target_return": target_rew},
                            stats,
                        )
                        reset_test(1)
                returns.append(ret)
                lengths.append(length)
            log_filename = eval_save_dir + f"/test_stats.p"
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
    num_agents = 4
    models = []
    for i in range(num_agents):
        models.append(load_model(model_path))
    env = create_env()
    def reset_envs(
        envs,
        test_case_fn,
        test_case_args,
        test_case,
        num_agents,
        policies,
        policy,
        prev_agents,
    ):
        test_case_args["num_agents"] = num_agents
        test_case_args["prev_agents"] = prev_agents
        agents = test_case_fn(**test_case_args)
        for env in envs:
            env.unwrapped.plot_policy_name = policy
            env.set_agents(agents)
            init_obs = env.reset()
            env.unwrapped.test_case_index = test_case
            agents.append(agents.pop(0))    # rotate agents 1 to left
        return init_obs, agents
    # model_evaluation(env, models, model_path)

model_path = DTConf['test']['model_path']
seed = DTConf.getint('test', 'seed')
multi = DTConf.getboolean('test', 'multi')

# np.random.seed(seed)
# torch.manual_seed(seed)
# random.seed(0)

if multi:
    test_multi(model_path)
else:
    env = create_env()
    model_evaluation(env, model_path)