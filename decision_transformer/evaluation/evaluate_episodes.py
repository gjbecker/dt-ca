import numpy as np
import torch
import os
### ADDED import grid representation
from decision_transformer.envs.gym_ca_data_gen.DATA.grid_representation import step_grid

def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
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

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length

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
    def episode_stats(episode_return, episode_length):
        generic_episode_stats = {
            "total_return": episode_return,
            "episode_length": episode_length,
        }

        agents = env.agents
        time_to_goal = np.array([a.t for a in agents])
        extra_time_to_goal = np.array(
            [a.t - a.straight_line_time_to_reach_goal for a in agents]
        )
        collision = np.array(np.any([a.in_collision for a in agents])).tolist()
        all_at_goal = np.array(np.all([a.is_at_goal for a in agents])).tolist()
        any_stuck = np.array(
            np.any([not a.in_collision and not a.is_at_goal for a in agents])
        ).tolist()
        outcome = (
            "collision" if collision else "all_at_goal" if all_at_goal else "stuck"
        )
        specific_episode_stats = {
            "num_agents": len(agents),
            "time_to_goal": time_to_goal,
            "total_time_to_goal": np.sum(time_to_goal),
            "extra_time_to_goal": extra_time_to_goal,
            "collision": collision,
            "all_at_goal": all_at_goal,
            "any_stuck": any_stuck,
            "outcome": outcome,
            "policies": [agent.policy.str for agent in agents],
        }

        # Merge all stats into a single dict
        return {**generic_episode_stats, **specific_episode_stats}
    
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
            stats = episode_stats(episode_return, episode_length)
            break
        
    return episode_return, episode_length, stats