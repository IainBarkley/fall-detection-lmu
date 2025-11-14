# lmu_classifier.py
# Terrence C. Stewart's LMU implementation in Nengo

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys,os
import nengo

sys.path.insert(0,'../networks')
from ldn import LDN

sys.path.insert(0,'../')
from utilities import generate_train_test_split

data_dir = '../fall_detection_data/merged_processed/'
subject_file = 'SA06-merged.csv'

# LDN parameters
size_in = 9         # 9 features of the fall detection data
theta = 1.
q = 42

# simulation parameters
dt = 0.001

# training dataset parameters
train_test_split = 0.5
chunk_size = 3600
scaling_factor = 100

# prediction parameters
decision_threshold = 0.5

# load data and generate the train/test split
ddf = pd.read_csv(os.path.join(data_dir,subject_file),index_col=0).drop(['TimeStamp(s)','FrameCounter'],axis=1)

chunk_indices = np.arange(0,ddf.shape[0],chunk_size)
train_chunk_indices = np.random.choice(chunk_indices,size = int(len(chunk_indices)*train_test_split))
test_chunk_indices = list(set(chunk_indices)-set(train_chunk_indices))

train_df = pd.DataFrame(columns=ddf.columns)
for idx in train_chunk_indices:
    train_df = pd.concat([train_df,ddf.iloc[idx:idx+chunk_size,:]],axis=0)

test_df = pd.DataFrame(columns=ddf.columns)
for idx in test_chunk_indices:
    test_df = pd.concat([test_df,ddf.iloc[idx:idx+chunk_size,:]],axis=0)

train_xs = train_df[['AccX','AccY','AccZ','GyrX','GyrY','GyrZ','EulerX','EulerY','EulerZ']].to_numpy()
lmu_train_xs = LDN(theta=theta, q=q, size_in=size_in).apply(train_xs)
train_ys = train_df[['Fall/No Fall']].to_numpy() * scaling_factor
test_xs = test_df[['AccX','AccY','AccZ','GyrX','GyrY','GyrZ','EulerX','EulerY','EulerZ']].to_numpy()
test_ys = test_df[['Fall/No Fall']].to_numpy() * scaling_factor

model = nengo.Network()
with model:

    def stim_func(t):
        index = int(t/dt)-1
        #return test_xs[index,:]
        return train_xs[index,:]


    stim = nengo.Node( size_out = size_in, output = stim_func )

    ldn = nengo.Node( LDN( theta = theta, q = q, size_in = size_in))
    nengo.Connection(stim, ldn, synapse=None)

    neurons = nengo.Ensemble(n_neurons=1000, dimensions=q*size_in, neuron_type=nengo.LIFRate())
    nengo.Connection(ldn, neurons, synapse = None)

    # initialize the network with the training data
    category = nengo.Ensemble(n_neurons=1000,dimensions=1,radius=100, neuron_type=nengo.Direct())
    nengo.Connection(neurons, category, eval_points=lmu_train_xs, function=train_ys, synapse = None)


    p_stim = nengo.Probe(stim)
    p_ldn = nengo.Probe(ldn)
    p_category = nengo.Probe(category, synapse=0.01)

sim = nengo.Simulator(model,dt=dt)
with sim:
    #sim.run(test_df.shape[0]*dt)
    sim.run(train_df.shape[0]*dt)

fig,ax = plt.subplots(1,1)
ax.plot(sim.trange(), sim.data[p_category], label='Predicted')
#ax.plot(sim.trange(), test_ys.flatten(), label='True')
ax.plot(sim.trange(), train_ys.flatten(), label='True')
ax.legend()
plt.show()

predictions = np.where(train_ys.flatten() > np.max(train_ys.flatten())*decision_threshold, 1, 0)
print(predictions.dtype)
print(train_ys.flatten().dtype)

tn, fp, fn, tp = confusion_matrix( train_ys, predictions.flatten() ).ravel()

for performance_metric, number in zip(('True Negatives','False Positives','False Negatives','True Positives'),(tn, fp, fn, tp)):
    print('{}: {}'.format(performance_metric,number))

fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
ax1.plot(sim.trange(), train_ys.flatten(), label='True')
ax2.plot(sim.trange(), sim.data[p_category]/scaling_factor, label='Predicted')
ax2.plot(sim.trange(),predictions)

ax1.legend()
ax2.legend()
plt.show()
