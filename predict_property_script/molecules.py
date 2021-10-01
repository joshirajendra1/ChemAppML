from megnet.models import MEGNetModel
import numpy as np
from operator import itemgetter
import json

def get_graph_from_doc(doc):
    """
    Convert a json document into a megnet graph
    """
    atom = [i['atomic_num'] for i in doc['atoms']]

    index1_temp = [i['a_idx'] for i in doc['atom_pairs']]
    index2_temp = [i['b_idx'] for i in doc['atom_pairs']]
    bond_temp = [i['spatial_distance'] for i in doc['atom_pairs']]

    index1 = index1_temp + index2_temp
    index2 = index2_temp + index1_temp
    bond = bond_temp + bond_temp
    sort_key = np.argsort(index1)
    it = itemgetter(*sort_key)

    index1 = it(index1)
    index2 = it(index2)
    bond = it(bond)
    graph = {'atom': atom, 'bond': bond, 'index1': index1, 'index2': index2, 'state': [[0, 0]]}
    return graph
    
# load an example qm9 document
with open('000001.json', 'r') as f:
    doc = json.load(f)
# convert to a graph
graph = get_graph_from_doc(doc)


# all target names
names = ['mu', 'alpha', 'HOMO', 'LUMO', 'gap', 'R2', 'ZPVE', 'U0', 'U', 'H', 'G', 'Cv', 'omega1']


y_pred = []
y_true = []

print('*** Result Comparisons ***')
print('Target\tMEGNet\tQM9')

for i in names:
    model = MEGNetModel.from_file('megnet/mvl_models/qm9-2018.6.1/' + i+'.hdf5')
    pred = model.predict_graph(graph)
    y_pred.append(pred)
    y_true.append(doc['mol_info'][i])
    print('%s\t%.3f\t%.3f' %(i, y_pred[-1], float(y_true[-1])))


