import pickle
from MCTS_chemisty import MCTS
from sampling import sampling_node

with open('search_space_pretrain', 'rb') as file:
    search_space = pickle.load(file)

with open('data/mosi_test', 'rb') as file:
    dataset = pickle.load(file)
validation = dict(list(dataset.items())[-10000:])
# validation = dataset

agent = MCTS(search_space, 5, 12)

for k, v in validation.items():
    agent.ROOT.validation[k] = v
agent.ROOT.bag = agent.ROOT.validation.copy()           
agent.predict_nodes('mean')
list = []
for node in agent.nodes:
    list.append(len(node.validation))

print(list)
nodes = [0, 1, 2, 3, 12, 13, 14, 15]
sampling_node(agent, nodes, dataset, 1)
    