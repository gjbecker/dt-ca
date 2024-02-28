import numpy as np
import torch
import os
import itertools
### ADDED import grid representation
from decision_transformer.envs.gym_ca.DATA.grid_representation import step_grid
from decision_transformer.envs.gym_ca.gym_collision_avoidance.envs.util import l2norm

def episode_stats(agents, episode_return, episode_length):
    generic_episode_stats = {
        "total_return": episode_return,
        "episode_length": episode_length,
    }

    time_to_goal = np.array([a.t for a in agents])
    extra_time_to_goal = np.array(
        [a.t - a.straight_line_time_to_reach_goal for a in agents]
    )
    collisions, at_goal, stuck = 0 ,0, 0
    for a in agents:
        if a.in_collision:
            collisions += 1
        elif a.is_at_goal:
            at_goal += 1
        else:
            stuck += 1
    collision = np.array(np.any([a.in_collision for a in agents])).tolist()
    all_at_goal = np.array(np.all([a.is_at_goal for a in agents])).tolist()
    any_stuck = np.array(
        np.any([not a.in_collision and not a.is_at_goal for a in agents])
    ).tolist()
    outcome = (
        "collision" if collision else "all_at_goal" if all_at_goal else "stuck"
    )
    dt_outcome = 'at_goal' if agents[0].is_at_goal else 'collision' if agents[0].in_collision else 'stuck'
    specific_episode_stats = {
        "num_agents": len(agents),
        "time_to_goal": time_to_goal,
        "total_time_to_goal": np.sum(time_to_goal),
        "extra_time_to_goal": extra_time_to_goal,
        "%_collisions": collisions/len(agents),
        "%_at_goal": at_goal/len(agents),
        "%_stuck": stuck/len(agents),
        "DT outcome": dt_outcome,
        "outcome": outcome,
        "policies": [agent.policy.str for agent in agents],
    }

    # Merge all stats into a single dict
    return {**generic_episode_stats, **specific_episode_stats}

### REMOVED evaluate_episode and evaluate_episode_RTG

### ADDED evaluation function for grid
def evaluate_episode_rtg_grid(
        env,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        RES=512,
    ):

    model.eval()
    model.to(device=device)

    try:
        state_mean = torch.from_numpy(state_mean).to(device=device)
        state_std = torch.from_numpy(state_std).to(device=device)
    except:
        state_mean = torch.zeros((1, 3, RES, RES)).to(device=device, dtype=torch.float32)
        state_std = torch.ones((1, 3, RES, RES)).to(device=device, dtype=torch.float32)
    
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    radii, goals, pols, state, acts, reward = ([] for _ in range(6))
    for i, agent in enumerate(env.agents):
        radii.append(agent.radius)
        goals.append([agent.goal_global_frame[0], agent.goal_global_frame[1]])
        pols.append(agent.policy.str)
        state.append([agent.pos_global_frame[0], agent.pos_global_frame[1], agent.heading_global_frame])
        acts.append([agent.past_actions[0][0].copy(), agent.past_actions[0][1].copy()])
    data = {
        'radii': radii, 
        'states': state, 
        'actions': acts,
        'rewards': reward,
        'goals': goals,
        'policies': pols
        }
    obs = step_grid(data, RES)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(obs).reshape(1,-1,3,RES,RES).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        # print(f'Evaluating step: {t} / {max_ep_len}', end='\r')
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        action = model.get_action(
            (states - state_mean) / state_std,
            actions.to(device=device, dtype=torch.float32),
            rewards.to(device=device, dtype=torch.float32),
            target_return.to(device=device, dtype=torch.float32),
            timesteps.to(device=device, dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, rew, done, truncated, info = env.step([action])
        radii, goals, pols, state, acts, reward = ([] for _ in range(6))
        for i, agent in enumerate(env.agents):
            radii.append(agent.radius)
            goals.append([agent.goal_global_frame[0], agent.goal_global_frame[1]])
            pols.append(agent.policy.str)
            reward.append(rew)
            state.append([agent.pos_global_frame[0], agent.pos_global_frame[1], agent.heading_global_frame])
            acts.append([agent.past_actions[0][0].copy(), agent.past_actions[0][1].copy()])
        data = {
            'radii': radii, 
            'states': state, 
            'actions': acts,
            'rewards': reward,
            'goals': goals,
            'policies': pols
            }
        obs = step_grid(data, RES)

        cur_state = torch.from_numpy(obs).to(device=device).reshape(1, -1, 3, RES, RES)
        states = torch.cat([states, cur_state], dim=1)
        rewards[-1] = rew[0]
        
        if mode != 'delayed':
            pred_return = target_return[0,-1] - (rew[0]/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += rew[0]
        episode_length += 1

        if done:
            stats = episode_stats(env.agents, episode_return, episode_length)
            break
        
    return episode_return, episode_length, stats

def evaluate_episode_rtg_d4rl(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):
    model.eval()
    model.to(device=device)

    try:
        state_mean = torch.from_numpy(state_mean).to(device=device)
        state_std = torch.from_numpy(state_std).to(device=device)
    except:
        state_mean = torch.zeros((1, state_dim)).to(device=device, dtype=torch.float32)
        state_std = torch.ones((1, state_dim)).to(device=device, dtype=torch.float32)
    
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    state = env.reset()
    
    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state[0][0]).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        # print(f'Evaluating step: {t} / {max_ep_len}', end='\r')
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        action = model.get_action(
            (states - state_mean) / state_std,
            actions.to(device=device, dtype=torch.float32),
            rewards.to(device=device, dtype=torch.float32),
            target_return.to(device=device, dtype=torch.float32),
            timesteps.to(device=device, dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, rew, done, truncated, info = env.step([action])

        cur_state = torch.from_numpy(state[0]).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = rew[0]
        
        if mode != 'delayed':
            pred_return = target_return[0,-1] - (rew[0]/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += rew[0]
        episode_length += 1

        if done:
            stats = episode_stats(env.agents, episode_return, episode_length)
            break
        
    return episode_return, episode_length, stats

def test_episode_rtg_grid(
        env,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        RES=512,
    ):
    
    model.eval()
    model.to(device=device)

    try:
        state_mean = torch.from_numpy(state_mean).to(device=device)
        state_std = torch.from_numpy(state_std).to(device=device)
    except:
        state_mean = torch.zeros((1, 3, RES, RES)).to(device=device, dtype=torch.float32)
        state_std = torch.ones((1, 3, RES, RES)).to(device=device, dtype=torch.float32)
    
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    radii, goals, pols, state, acts, reward = ([] for _ in range(6))
    for i, agent in enumerate(env.agents):
        radii.append(agent.radius)
        goals.append([agent.goal_global_frame[0], agent.goal_global_frame[1]])
        pols.append(agent.policy.str)
        state.append([agent.pos_global_frame[0], agent.pos_global_frame[1], agent.heading_global_frame])
        acts.append([agent.past_actions[0][0].copy(), agent.past_actions[0][1].copy()])
    data = {
        'radii': radii, 
        'states': state, 
        'actions': acts,
        'rewards': reward,
        'goals': goals,
        'policies': pols
        }
    obs = step_grid(data, RES)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(obs).reshape(1,-1,3,RES,RES).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0
    turn_rate = [0]
    turn_stats = True
    for t in range(max_ep_len):
        # print(f'Evaluating step: {t} / {max_ep_len}', end='\r')
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        action = model.get_action(
            (states - state_mean) / state_std,
            actions.to(device=device, dtype=torch.float32),
            rewards.to(device=device, dtype=torch.float32),
            target_return.to(device=device, dtype=torch.float32),
            timesteps.to(device=device, dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, rew, done, truncated, info = env.step([action])
        radii, goals, pols, state, acts, reward = ([] for _ in range(6))
        for i, agent in enumerate(env.agents):
            radii.append(agent.radius)
            goals.append([agent.goal_global_frame[0], agent.goal_global_frame[1]])
            pols.append(agent.policy.str)
            reward.append(rew)
            state.append([agent.pos_global_frame[0], agent.pos_global_frame[1], agent.heading_global_frame])
            acts.append([agent.past_actions[0][0].copy(), agent.past_actions[0][1].copy()])
        data = {
            'radii': radii, 
            'states': state, 
            'actions': acts,
            'rewards': reward,
            'goals': goals,
            'policies': pols
            }
        obs = step_grid(data, RES)
        if turn_stats:
            dist_btwn_nearest_agent = [np.inf for _ in env.agents]
            agent_inds = list(range(len(env.agents)))
            agent_pairs = list(itertools.combinations(agent_inds, 2))
            for i, j in agent_pairs:
                dist_btwn = l2norm(
                    env.agents[i].pos_global_frame,
                    env.agents[j].pos_global_frame,
                )
                combined_radius = env.agents[i].radius + env.agents[j].radius
                dist_btwn_nearest_agent[i] = min(
                    dist_btwn_nearest_agent[i], dist_btwn - combined_radius
                )
                dist_btwn_nearest_agent[j] = min(
                    dist_btwn_nearest_agent[j], dist_btwn - combined_radius
                )
            if dist_btwn_nearest_agent[0] <= 0.2:
                turn_rate_deg = abs(acts[0][1]) * 180 / np.pi   # convert to degrees
                turn_rate.append(turn_rate_deg)

        cur_state = torch.from_numpy(obs).to(device=device).reshape(1, -1, 3, RES, RES)
        states = torch.cat([states, cur_state], dim=1)
        rewards[-1] = rew[0]
        
        if mode != 'delayed':
            pred_return = target_return[0,-1] - (rew[0]/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += rew[0]
        episode_length += 1

        if done:
            stats = episode_stats(env.agents, episode_return, episode_length)
            break
        
    return episode_return, episode_length, stats

def test_episode_rtg_d4rl(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        RES=512,
    ):
    
    model.eval()
    model.to(device=device)

    try:
        state_mean = torch.from_numpy(state_mean).to(device=device)
        state_std = torch.from_numpy(state_std).to(device=device)
    except:
        state_mean = torch.zeros((1, state_dim)).to(device=device, dtype=torch.float32)
        state_std = torch.ones((1, state_dim)).to(device=device, dtype=torch.float32)
    
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    state = env.reset()
    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state[0][0]).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0
    turn_rate = [0]
    turn_stats = False
    for t in range(max_ep_len):
        # print(f'Evaluating step: {t} / {max_ep_len}', end='\r')
        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        action = model.get_action(
            (states - state_mean) / state_std,
            actions.to(device=device, dtype=torch.float32),
            rewards.to(device=device, dtype=torch.float32),
            target_return.to(device=device, dtype=torch.float32),
            timesteps.to(device=device, dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, rew, done, truncated, info = env.step([action])

        if turn_stats:
            dist_btwn_nearest_agent = [np.inf for _ in env.agents]
            agent_inds = list(range(len(env.agents)))
            agent_pairs = list(itertools.combinations(agent_inds, 2))
            for i, j in agent_pairs:
                dist_btwn = l2norm(
                    env.agents[i].pos_global_frame,
                    env.agents[j].pos_global_frame,
                )
                combined_radius = env.agents[i].radius + env.agents[j].radius
                dist_btwn_nearest_agent[i] = min(
                    dist_btwn_nearest_agent[i], dist_btwn - combined_radius
                )
                dist_btwn_nearest_agent[j] = min(
                    dist_btwn_nearest_agent[j], dist_btwn - combined_radius
                )
            if dist_btwn_nearest_agent[0] <= 0.2:
                turn_rate_deg = abs(env.agents[0].past_actions[0][1].copy()) * 180 / np.pi   # convert to degrees
                turn_rate.append(turn_rate_deg)

        cur_state = torch.from_numpy(state[0]).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = rew[0]
        
        if mode != 'delayed':
            pred_return = target_return[0,-1] - (rew[0]/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += rew[0]
        episode_length += 1

        if done:
            stats = episode_stats(env.agents, episode_return, episode_length)
            break
        
    return episode_return, episode_length, stats