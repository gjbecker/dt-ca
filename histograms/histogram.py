import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import os


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

dataset_path = '/work/flemingc/gjbecker/decision-transformer/gym/decision_transformer/envs/gym_ca_data_gen/DATA/datasets/RVO_circle_agent_5000.pkl'
dataset = pd.read_pickle(dataset_path)
print(f'Creating histograms for {dataset.split("/")[-1]}')
save_dir = os.path.dirname(__file__) + dataset_path.split('/')[-1].split('_ag')[0]
os.makedirs(save_dir, exist_ok=True)

returns = []
actions = []
head = []
speed = []
i=0

for ep in dataset:
    # if i == 1000:
    #     break
    for ego in np.arange(len(ep['radii'])):
        returns += list(discount_cumsum(np.array(ep['rewards'][ego]), gamma=1))
        for act in ep['actions'][ego]:
            speed.append(act[0])
            head.append(act[1])
    i += 1

# print(len(np.array(returns)))
# print(len(np.array(speed)))
# print(len(np.array(head)))
# print(min(speed))

plt.hist(x=returns, bins=50)
plt.title(f'Frequency Distribution of RTG --- {dataset_path.split("/")[-1].split("_ag")[0]}')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.savefig(save_dir + '/returns.png')
plt.xlim(left=-10)
plt.savefig(save_dir + '/returns_trimmed.png')
print('Saved returns plot')

plt.hist2d(x=speed, y=head, bins=50)
plt.title(f'Frequency Distribution of Actions --- {dataset_path.split("/")[-1].split("_ag")[0]}')
plt.xlabel('Speed')
plt.ylabel('Del Heading')
plt.savefig(save_dir + '/actions.png')
print('Saved actions 2D plot')