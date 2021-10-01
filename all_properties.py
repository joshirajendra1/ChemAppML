from flask import Flask, redirect, url_for, request
app = Flask(__name__)

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



@app.route('/success/<name>')
#def success(name):
#   return 'welcome %s' % name

def success(name):
    names = ['mu', 'alpha', 'HOMO', 'LUMO', 'gap', 'R2', 'ZPVE', 'U0', 'U', 'H', 'G', 'Cv', 'omega1']
    
    y_pred = []
    y_true = []
    predicted = {}
    
    for i in names:
        model = MEGNetModel.from_file('../simple_app/megnet/mvl_models/qm9-2018.6.1/' + i+'.hdf5')
        pred = model.predict_graph(graph)
        y_pred.append(pred)
        y_true.append(doc['mol_info'][i])
        predicted[i] = [str(list(pred)[0]), str(float(doc['mol_info'][i]))]        
    return predicted



@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['nm']
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))


if __name__ == '__main__':
   app.run(debug = True)
