import pickle
import pandas as pd
import os
import sys
import pprint

dir = os.path.dirname(os.path.realpath(__file__))

file = input(f'Enter file location: {dir}/ ')

file = 'RVO_4/RVO_4_agents/20231018-2340/stats.p'
path = os.path.join(dir, file)

stats = pd.read_pickle(path)
print('\n',path)
print('\n', list(stats.columns))
print('\n', stats[stats['target_return']==1]['eval_episode']['total_return']['outcome'].tail)

# stats = []
# with open(path, 'rb') as fr:
#     try:
#         while True:
#             stats.append(pickle.load(fr))
#     except EOFError:
#         pass
# print(stats)