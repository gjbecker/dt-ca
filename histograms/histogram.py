import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import os
from statistics import mode


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

dataset_path = '/work/flemingc/gjbecker/dt-ca/decision_transformer/envs/gym_ca_data_gen/DATA/datasets/circle8_4_agent_5000.pkl'
dataset = pd.read_pickle(dataset_path)
print(f'Creating histograms for {dataset_path.split("/")[-1]}')
save_dir = os.path.dirname(__file__) + "/" + dataset_path.split('/')[-1].split('_ag')[0]
os.makedirs(save_dir, exist_ok=True)

returns = []
actions = []
head = []
speed = []
i=0
rew_4 = []
rew_6 = []
rew_8 = []
for ep in dataset:
    # if i == 1000:
    #     break
    agents = len(ep['radii'])
    for ego in np.arange(agents):
        returns += list(discount_cumsum(np.array(ep['rewards'][ego]), gamma=1))
        # if agents == 4:
        #     rew_4 += list(discount_cumsum(np.array(ep['rewards'][ego]), gamma=1))
        # elif agents == 6:
        #     rew_6 += list(discount_cumsum(np.array(ep['rewards'][ego]), gamma=1))
        # else:
        #     rew_8 += list(discount_cumsum(np.array(ep['rewards'][ego]), gamma=1))

        for act in ep['actions'][ego]:
            speed.append(act[0])
            head.append(act[1])
    i += 1

# print(len(np.array(returns)))
# print(len(np.array(speed)))
# print(len(np.array(head)))
# print(min(speed))
# print(f'Max speed: {max(speed)}')
print(f'Head min: {min(head)}')
print(f'Head max: {max(head)}')
print(f'Reward Max: {max(returns)} | Mean {np.mean(returns)} | Median {np.median(returns)} | Mode {mode(returns)}')

plt.hist(x=returns, bins=50, label='All Agent')
plt.title(f'Frequency Distribution of RTG --- {dataset_path.split("/")[-1].split("_ag")[0]}')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.savefig(save_dir + '/returns.png')
plt.xlim(left=-10)
plt.savefig(save_dir + '/returns_trimmed.png')

# plt.clf()
# plt.hist(x=rew_4, bins=50, label='4 Agent')
# plt.title(f'Frequency Distribution of RTG --- {dataset_path.split("/")[-1].split("_ag")[0]} | 4 agent')
# plt.xlabel('Returns')
# plt.ylabel('Frequency')
# plt.savefig(save_dir + '/returns4.png')
# plt.clf()
# plt.hist(x=rew_6, bins=50, label='6 Agent')
# plt.title(f'Frequency Distribution of RTG --- {dataset_path.split("/")[-1].split("_ag")[0]}  | 6 agent')
# plt.xlabel('Returns')
# plt.ylabel('Frequency')
# plt.savefig(save_dir + '/returns6.png')
# plt.clf()
# plt.hist(x=rew_8, bins=50, label='8 Agent')
# plt.title(f'Frequency Distribution of RTG --- {dataset_path.split("/")[-1].split("_ag")[0]}  | 8 agent')
# plt.xlabel('Returns')
# plt.ylabel('Frequency')
# plt.savefig(save_dir + '/returns8.png')
# plt.clf()
# plt.hist(x=returns, bins=50, label='All Agent')
# plt.hist(x=rew_4, bins=50, label='4 Agent')
# plt.hist(x=rew_6, bins=50, label='6 Agent')
# plt.hist(x=rew_8, bins=50, label='8 Agent')
# plt.title(f'Frequency Distribution of RTG --- {dataset_path.split("/")[-1].split("_ag")[0]}')
# plt.legend()
# plt.savefig(save_dir + '/returns_all.png')
print('Saved returns plot')
plt.clf()
plt.hist2d(x=speed, y=head, bins=50)
plt.title(f'Frequency Distribution of Actions --- {dataset_path.split("/")[-1].split("_ag")[0]}')
plt.xlabel('Speed')
plt.ylabel('Del Heading')
plt.savefig(save_dir + '/actions.png')
print('Saved actions 2D plot')