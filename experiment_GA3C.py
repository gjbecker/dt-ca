import gym
import numpy as np
import pandas as pd
import torch
import wandb
import argparse
import pickle as pkl
import random
import sys, os, datetime, time
from copy import deepcopy
import DT_GA3C_config as DTconfig
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg_ca
from decision_transformer.models.decision_transformer import DecisionTransformer
# from decision_transformer.models.mlp_bc import MLPBCModel             # BC Model
# from decision_transformer.training.act_trainer import ActTrainer      # BC Trainer
from decision_transformer.training.seq_trainer import SequenceTrainer


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{env_name}{dataset.split("_")[2]}-{dataset.split("_agent")[0]}'       # REMOVED exp_prefix
    if DTconfig.resume:
        model_id = DTconfig.load_path.split('/')[-1].split('.')[0]
    else:
        model_id = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    exp_prefix = f'{group_name}-{model_id}'

    ### REMOVED other gym environments, original implementation in experiment.py
    if env_name[-4:] == 'GA3C':
        # ADDED check for gym_collision_avoidance module 
        from gym_collision_avoidance.experiments.src.env_utils import create_env, store_stats
        from gym_collision_avoidance.envs import Config
        import gym_collision_avoidance.envs.test_cases as tc
        
        num_agents = DTconfig.num_agents
        policies = DTconfig.policies
        if DTconfig.varying_agents == True:
            pad_data = True
            assert(DTconfig.max_num_agents<=num_agents), 'Max number of agents too low'
            Config.MAX_NUM_AGENTS_IN_ENVIRONMENT = DTconfig.max_num_agents
            Config.MAX_NUM_AGENTS_TO_SIM = DTconfig.max_num_agents
            Config.MAX_NUM_OTHER_AGENTS_OBSERVED = DTconfig.max_num_agents - 1
            eval_save_dir = os.path.dirname(os.path.realpath(__file__)) + f"/model_eval/{dataset.split('/')[0]}/{policies}_{num_agents}-{DTconfig.max_num_agents}_agents/{env_name}/{model_id}/"
        else:    
            pad_data = False    
            Config.MAX_NUM_AGENTS_IN_ENVIRONMENT = num_agents
            Config.MAX_NUM_AGENTS_TO_SIM = num_agents
            Config.MAX_NUM_OTHER_AGENTS_OBSERVED = num_agents - 1
            eval_save_dir = os.path.dirname(os.path.realpath(__file__)) + f"/model_eval/{dataset.split('/')[0]}/{policies}_{num_agents}_agents/{env_name}/{model_id}/"
        os.makedirs(eval_save_dir, exist_ok=True)
        num_actions = int(dataset.split('_')[2])
        Config.SAVE_EPISODE_PLOTS = True
        Config.STATES_IN_OBS = ['num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', 'other_agents_states']
        Config.setup_obs()

        env = create_env()
        env_targets = DTconfig.env_targets

        if env_name == 'cont_GA3C':
            dataset_act = 'c_actions'
            act_dim = env.action_space.shape[0]
        elif env_name == 'disc_GA3C':
            dataset_act = 'd_actions'
            act_dim = num_actions
        else:
            assert(0), 'invalid env_name'

        # test_cases = []
        # tc_args = DTconfig.test_case_args
        # for _ in range(DTconfig.max_iters*DTconfig.num_eval_episodes):  
        #     test_case = tc.tc.generate_rand_test_case_multi(
        #         num_agents=DTconfig.num_agents, side_length=5, speed_bnds=[0.5, 2.0], radius_bnds=[0.5,0.5]
        #     )
        #     test_cases.append(test_case)
        

        test_cases = pd.read_pickle(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            + f'/gym_ca/gym_collision_avoidance/envs/test_cases/{num_agents}_agents_500_cases.p'
            )
        
        def reset_test(case_num, iter_num):
            def reset_env(env, agents, case_num, policy,):
                env.unwrapped.plot_policy_name = policy        
                env.set_agents(agents)
                env.reset()
                env.unwrapped.test_case_index = case_num

            ### Get agents from test case
            # test_ = test_cases[(DTconfig.num_eval_episodes*(iter_num-1))+case_num-1]
            # agents = tc.cadrl_test_case_to_agents(test_, tc_args['policies'], tc_args['policy_distr'])
            # agents[0].policy = tc.policy_dict['external']()   # set first agent policy to external and init
            if case_num > 20:
                policies = ['noncoop']*num_agents
            else:
                policies = [DTconfig.policies]*num_agents
            policies[0] = 'external'
            agents = tc.cadrl_test_case_to_agents(test_case=test_cases[case_num-1], policies=policies)

            reset_env(env, agents, case_num, policy='DT')
            
        max_ep_len = DTconfig.max_ep_len        
        scale = DTconfig.scale
    else:
        raise NotImplementedError

    ### REMOVED BC target modification
    state_dim = env.observation_space.shape[1]

    # load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), '../datasets_ca/GA3C', dataset)
    print('Loading data from \n  ' + dataset_path)
    with open(dataset_path, 'rb') as f:
        trajectories = pkl.load(f)

    mode = variant.get('mode', 'normal')

    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    if DTconfig.state_mean != None:
        state_mean = DTconfig.state_mean; state_std = DTconfig.state_std

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'State dim: {state_dim} | Action dim: {act_dim}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)
    # assert(0)
    K = variant['K']      
    gamma = DTconfig.gamma
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    ### REMOVED pct_traj

    ### REMOVED %BC Training Configuration
    sorted_inds = np.argsort(returns)  # lowest to highest

    ### REMOVED p_sample, do not want bias towards longer episodes

    def get_batch(batch_size=256, max_len=K):
        # batch_start = time.time()
        batch_inds = np.random.choice(
            np.arange(len(traj_lens)),
            size=batch_size,
            replace=True,
            # p=p_sample,  # reweights so we sample according to timesteps
        )
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            idx = int(sorted_inds[batch_inds[i]])
            ### ADDED state transformation to grid
            traj_obs = trajectories[idx]['observations']
            traj_acts = trajectories[idx][dataset_act]
            traj_rews = trajectories[idx]['rewards']
            si = random.randint(0, traj_rews.shape[0] - 1)
            if dataset_act == 'd_actions':
                act_idx = traj_acts[si:si+max_len]
                transformed_acts = []
                for a_idx in act_idx:
                    a_ = np.zeros((num_actions))
                    a_[a_idx] = 1
                    transformed_acts.append(a_)
                transformed_acts = np.array(transformed_acts)
            else:
                transformed_acts = traj_acts[si:si+max_len]
            # get sequences from dataset
            ### CHANGED Reshape parameters for state
            s.append(traj_obs[si:si + max_len].reshape(1, -1, state_dim))    # Linear Layer
            a.append(transformed_acts.reshape(1, -1, act_dim))
            r.append(traj_rews[si:si + max_len].reshape(1, -1, 1))
            ### DELETED Dones, not used in transformer

            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj_rews[si:], gamma=gamma)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            ### CHANGED Reshape parameters for state
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)   # Linear Layer
            ### REMOVED State normalization
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            ### REMOVED Dones
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale       # Scale is game dependent
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        ### REMOVED Dones
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        # print(f'Get batch time: {time.time() - batch_start}')
        return s, a, r, d, rtg, timesteps, mask
    

    def eval_episodes(target_rew):
        def fn(model, iteration):
            returns, lengths = [], []
            print()
            env.set_plot_save_dir(eval_save_dir+ f'/{target_rew}/')
            reset_test(1, iteration)
            df = pd.DataFrame()
            outcomes = []
            for x in range(num_eval_episodes):
                print(f'Eval episode: {x+1} / {num_eval_episodes}', end='\r')
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length, stats = evaluate_episode_rtg_ca(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                        df = store_stats(
                            df,
                            {"eval_episode": x+1, "target_return": target_rew},
                            stats,
                        )
                        outcomes.append(1 if (stats['DT outcome'] == 'at_goal') else 0)
                        reset_test(x+2, iteration)
                ### REMOVED BC Evaluation function
                ### ADDED Evaluation more statistics
                returns.append(ret)
                lengths.append(length)
            log_filename = eval_save_dir + f"/stats.p"
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
                f'target_{target_rew}_goal_%' : np.mean(outcomes)
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            gamma=gamma,
        )
    ### REMOVED BC Model
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    save_model_path = f"{os.path.dirname(os.path.realpath(__file__))}/models/{exp_prefix}.pt"
    def checkpoint_save(model, optimizer, scheduler, ckpt_dir, iter):
        ckpt_path = ckpt_dir+f'/{model_id}.pth'
        torch.save({
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        },
        ckpt_path)
        wandb.save(ckpt_path) # saves checkpoint to wandb
    
    start_iter = 0
    resume = False
    if log_to_wandb:
        if DTconfig.resume:
            model_id = DTconfig.load_path.split('/')[-1].split('.')[0]
            resume = 'must'
            checkpoint_load_path = DTconfig.load_path

        run = wandb.init(
            id=model_id,
            group=group_name,
            project='decision-transformer',
            entity='dt-collision-avoidance',
            resume=resume,
            config=variant
        )
        if wandb.run.resumed:
            wandb.restore(f'{model_id}.pth')
            checkpoint = torch.load(checkpoint_load_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iter = checkpoint['iter'] + 1
            print(f"Resuming {run.path} at iter {start_iter} / {variant['max_iters']}")
        wandb.watch(model)  # wandb has some bug


    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    ### REMOVED BC Trainer
    for iter in range(start_iter, variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=DTconfig.print_logs)
        
        if DTconfig.checkpoint:
            ckpt_dir = f"{os.path.dirname(os.path.realpath(__file__))}/models/{dataset.split('/')[0]}/"
            os.makedirs(ckpt_dir, exist_ok=True)
            checkpoint_save(model, optimizer, scheduler, ckpt_dir, iter)
        if log_to_wandb:
            wandb.log(outputs)

    if log_to_wandb:
        torch.save(model.state_dict(), save_model_path)
        artifact = wandb.Artifact(dataset.split('/')[0], 'model')
        artifact.add_file(save_model_path)
        run.log_artifact(artifact)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=DTconfig.env_name)
    parser.add_argument('--dataset', type=str, default=DTconfig.dataset)  
    parser.add_argument('--mode', type=str, default=DTconfig.mode)  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=DTconfig.k)
    ### REMOVED pct_traj
    parser.add_argument('--batch_size', type=int, default=DTconfig.batch_size)   
    parser.add_argument('--model_type', type=str, default=DTconfig.model_type)  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=DTconfig.embed_dim)
    parser.add_argument('--n_layer', type=int, default=DTconfig.n_layer)
    parser.add_argument('--n_head', type=int, default=DTconfig.n_head)
    parser.add_argument('--activation_function', type=str, default=DTconfig.activation_fn)
    parser.add_argument('--dropout', type=float, default=DTconfig.dropout)
    parser.add_argument('--learning_rate', '-lr', type=float, default=DTconfig.lr)
    parser.add_argument('--weight_decay', '-wd', type=float, default=DTconfig.wd)
    parser.add_argument('--warmup_steps', type=int, default=DTconfig.warmup)
    parser.add_argument('--num_eval_episodes', type=int, default=DTconfig.num_eval_episodes)   
    parser.add_argument('--max_iters', type=int, default=DTconfig.max_iters)       
    parser.add_argument('--num_steps_per_iter', type=int, default=DTconfig.num_steps_per_iter) 
    parser.add_argument('--device', type=str, default=DTconfig.device)  
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=DTconfig.wandb)
    
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
