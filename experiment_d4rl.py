import gym
import numpy as np
import pandas as pd
import torch
import wandb
import configparser
import argparse
import pickle as pkl
import random
import sys, os, datetime, time

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg_d4rl
from decision_transformer.models.decision_transformer_linear import DecisionTransformer
# from decision_transformer.models.mlp_bc import MLPBCModel             # BC Model
# from decision_transformer.training.act_trainer import ActTrainer      # BC Trainer
from decision_transformer.training.seq_trainer import SequenceTrainer

DTConf = configparser.ConfigParser()
DTConf.read('experiment.config')

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
    group_name = f'{env_name}-{dataset.split("_agent")[0]}'       # REMOVED exp_prefix
    if DTConf.getboolean('checkpoint', 'resume'):
        model_id = DTConf['checkpoint']['load_path'].split('/')[-1].split('.')[0]
    else:
        model_id = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    exp_prefix = f'{group_name}-{model_id}'

    ### REMOVED other gym environments, original implementation in experiment.py
    if env_name == 'fixed-d4rl':
        # ADDED check for gym_collision_avoidance module 
        from decision_transformer.envs.gym_ca.gym_collision_avoidance.experiments.src.env_utils import create_env, store_stats
        from decision_transformer.envs.gym_ca.gym_collision_avoidance.envs import Config
        Config.EVALUATE_MODE = True
        Config.SAVE_EPISODE_PLOTS = True
        Config.SHOW_EPISODE_PLOTS = True
        Config.DT = 0.1
        Config.PLOT_CIRCLES_ALONG_TRAJ = True
        env = create_env()
        try:
            num_agents = int(dataset.split('_')[1])
            policies = ['RVO']*num_agents
            policies[0] = 'external'

            test_cases = pd.read_pickle(
                os.path.dirname(os.path.realpath(__file__)) 
                + f'/decision_transformer/envs/gym_ca/gym_collision_avoidance/envs/test_cases/{num_agents}_agents_500_cases.p'
                )
            
            def reset_test(case_num):
                import decision_transformer.envs.gym_ca.gym_collision_avoidance.envs.test_cases as tc

                def reset_env(env, agents, case_num, policy,):
                    env.unwrapped.plot_policy_name = policy        
                    env.set_agents(agents)
                    init_obs = env.reset()
                    env.unwrapped.test_case_index = case_num
                    return init_obs, agents
                
                if case_num > 20:
                    policies = ['noncoop']*num_agents
                else:
                    policies = ['RVO']*num_agents
                policies[0] = 'external'

                agents = tc.cadrl_test_case_to_agents(test_case=test_cases[case_num-1], policies=policies)
                _, _ = reset_env(env, agents, case_num, policy='DT')

        except:
            print('Could not find gym_collision_avoidance module. Was it installed?')
            sys.exit()
        
        assert Config.MAX_NUM_AGENTS_IN_ENVIRONMENT == num_agents, f'{Config.MAX_NUM_AGENTS_IN_ENVIRONMENT} != {num_agents}'
            
        max_ep_len = DTConf.getint('env', 'max_ep_len')
        eval_save_dir = os.path.dirname(os.path.realpath(__file__)) + f"/model_eval/{dataset.split('_a')[0]}/{policies[1]}_{num_agents}_agents/{env_name}/{model_id}/"
        os.makedirs(eval_save_dir, exist_ok=True)
        
        scale = DTConf.getint('env', 'scale') 
        res = DTConf.getint('env', 'res')   # Change shape accordingly in decision_transformer.forward and update convnet linear dimension
    else:
        raise NotImplementedError

    ### REMOVED BC target modification
    state_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f'decision_transformer/envs/gym_ca/DATA/{dataset}'
    print('Loading data from ' + dataset_path)
    with open(dataset_path, 'rb') as f:
        trajectories = pkl.load(f)

    observations, actions, rewards, traj_lens, returns = [[] for _ in range(5)]
    state_dim_data, num_agents_data = trajectories['observations'][0].shape[2], trajectories['observations'][0].shape[1]
    act_dim_data = trajectories['actions'][0].shape[2]

    mode = variant.get('mode', 'normal')

    for i in range(len(trajectories['observations'])):
        obs = trajectories['observations'][i].reshape(-1, state_dim_data)[0::num_agents_data]
        observations.append(obs)
        act = trajectories['actions'][i].reshape(-1, act_dim_data)[0::num_agents_data]
        actions.append(act)
        rew = np.array(trajectories['rewards'][i].reshape(-1,1)[0::num_agents_data]).reshape(-1)
        rewards.append(rew)
        traj_lens.append(len(trajectories['observations'][i]))
        returns.append(np.array(rew).sum())
 
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    env_targets = []
    env_targets.append(round(max(returns), 2))
    env_targets.append(round(np.mean(returns), 2))
    for tar in env_targets:
        os.makedirs(eval_save_dir + f'/{tar}/', exist_ok=True)
    print('Data loaded successfully!')

    ### REMOVED Input Normalization
    state_mean, state_std = DTConf.getint('env', 'state_mean'), DTConf.getint('env', 'state_std')

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'State dim: {state_dim} | Action dim: {act_dim}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']      
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
            traj_obs = np.array(observations[idx])
            traj_acts = np.array(actions[idx])
            traj_rews = np.array(rewards[idx])
            si = random.randint(0, traj_rews.shape[0] - 1)
            # get sequences from dataset
            ### CHANGED Reshape parameters for state
            s.append(traj_obs[si:si + max_len].reshape(1, -1, state_dim))    # Linear Layer
            a.append(traj_acts[si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj_rews[si:si + max_len].reshape(1, -1, 1))
            ### DELETED Dones, not used in transformer

            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj_rews[si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
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
        def fn(model):
            returns, lengths = [], []
            print()
            env.set_plot_save_dir(eval_save_dir+ f'/{target_rew}/')
            reset_test(1)
            df = pd.DataFrame()
            for x in range(num_eval_episodes):
                # print(f'Eval episode: {x+1} / {num_eval_episodes}', end='\r')
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length, stats = evaluate_episode_rtg_d4rl(
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
                        reset_test(x+2)
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
        if DTConf.getboolean('checkpoint', 'resume'):
            model_id = DTConf['checkpoint']['load_path'].split('/')[-1].split('.')[0]
            resume = 'must'
            checkpoint_load_path = DTConf['checkpoint']['load_path']

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
        if not wandb.run.resumed:
            wandb.log({'res':res})


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
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=DTConf.getboolean('env', 'print_logs'))
        
        if DTConf.getboolean('checkpoint', 'checkpoint'):
            ckpt_dir = f"{os.path.dirname(os.path.realpath(__file__))}/models/{dataset.split('_a')[0]}/"
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
    parser.add_argument('--env', type=str, default=DTConf['env']['env_name'])
    parser.add_argument('--dataset', type=str, default=DTConf['env']['dataset'])  
    parser.add_argument('--mode', type=str, default=DTConf['env']['mode'])  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=DTConf.getint('model', 'k'))
    ### REMOVED pct_traj
    parser.add_argument('--batch_size', type=int, default=DTConf.getint('model', 'batch_size'))   
    parser.add_argument('--model_type', type=str, default=DTConf['model']['model_type'])  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=DTConf.getint('model', 'embed_dim'))
    parser.add_argument('--n_layer', type=int, default=DTConf.getint('model', 'n_layer'))
    parser.add_argument('--n_head', type=int, default=DTConf.getint('model', 'n_head'))
    parser.add_argument('--activation_function', type=str, default=DTConf['model']['activation_fn'])
    parser.add_argument('--dropout', type=float, default=DTConf.getfloat('model', 'dropout'))
    parser.add_argument('--learning_rate', '-lr', type=float, default=DTConf.getfloat('model', 'lr'))
    parser.add_argument('--weight_decay', '-wd', type=float, default=DTConf.getfloat('model', 'wd'))
    parser.add_argument('--warmup_steps', type=int, default=DTConf.getint('model', 'warmup'))
    parser.add_argument('--num_eval_episodes', type=int, default=DTConf.getint('model', 'num_eval_episodes'))   
    parser.add_argument('--max_iters', type=int, default=DTConf.getint('model', 'max_iters'))       
    parser.add_argument('--num_steps_per_iter', type=int, default=DTConf.getint('model', 'num_steps_per_iter')) 
    parser.add_argument('--device', type=str, default=DTConf['model']['device'])  
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=DTConf['model']['wandb'])
    
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
