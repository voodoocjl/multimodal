import pickle
import os
import json
import torch
from MCTS_chemisty import MCTS

state_path = 'results/agent'
files = os.listdir(state_path)
if files:
    # files.sort(key=lambda x: os.path.getmtime(os.path.join(state_path, x)))
    # # node_path = os.path.join(state_path, files[-1])
    for file in files:
        node_path = os.path.join(state_path, file)
        # node_path = 'results/mcts_agent_4000'
        with open(node_path, 'rb') as json_data:
            agent = pickle.load(json_data)              
        agent.print_tree()
        print('CP = ', agent.Cp)
        print(agent.nodes[0].classifier.model)
