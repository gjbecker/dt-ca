import numpy as np
import collections
import pickle as pkl
import h5py
import argparse
import os

parser = argparse.ArgumentParser(prog='Dataset conversion', description='Split data by episode')
parser.add_argument('--filename', '-f', type=str)
args = parser.parse_args()

filename = args.filename
if not os.path.isfile(filename):
    assert(0), f'{filename} does not exist'
name = filename.split('.')[0]

dataset = {}

with h5py.File(filename, 'r') as f:
    for key in list(f.keys()):
        if key == 'metadata':
            continue
        dataset[key] = f[key][()]

N = dataset['rewards'].shape[0]
data_ = collections.defaultdict(list)

use_timeouts = False
if 'timeouts' in dataset:
    use_timeouts = True

episode_step = 0
paths = []
for i in range(N):
    done_bool = bool(dataset['terminals'][i])
    if use_timeouts:
        final_timestep = dataset['timeouts'][i]
    else:
        final_timestep = False
    for k in ['observations', 'c_actions', 'd_actions', 'rewards', 'terminals']:
        data_[k].append(dataset[k][i])
    if done_bool or final_timestep:
        episode_step = 0
        episode_data = {}
        for k in data_:
            episode_data[k] = np.array(data_[k])
        paths.append(episode_data)
        data_ = collections.defaultdict(list)
    episode_step += 1

returns = np.array([np.sum(p['rewards']) for p in paths])
num_samples = np.sum([p['rewards'].shape[0] for p in paths])
print(f'Number of samples collected: {num_samples}')
print(f'Trajectory returns: mean = {np.mean(returns):.2f}, std = {np.std(returns):.2f}, max = {np.max(returns):.2f}, min = {np.min(returns):.2f}')

with open(f'{name}.pkl', 'wb') as f:
    pkl.dump(paths, f)
