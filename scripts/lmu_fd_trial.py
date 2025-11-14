from sklearn.metrics import confusion_matrix,roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys,os

import nengo
import pytry

sys.path.insert(0,'../networks')
from ldn import LDN

class LMUFallTrial(pytry.Trial):
    def params(self):
        self.param('Training strategy', training_strategy = 'within-subject')
        self.param('Subject to train on', subject_to_train_on = 'SA07' )
        
        self.param('Directory of training data', training_data_dir = '../fall_detection_data/not_merged_processed/')
        self.param('Train/Test split', train_test_split = 0.5)
        self.param('Label scaling factor', scaling_factor = 100.)
        self.param('Simulation time constant', dt = 0.001)
        self.param('List of features of the data to include', included_features = 'All')
        self.param('Normalization method', normalization_method = 'Min/Max')
        
        # Network parameters
        self.param('Window size of the LDN', theta = 1.)
        self.param('Dimensionality of the LDN', q = 42)
        self.param('Number of neurons in the hidden layer', hl_neurons = 1000)
        self.param('Radius of representation in hidden layer', hl_radius = 1.)
        self.param('Type of neuron to use in the hidden layer', hl_neuron_type = nengo.RectifiedLinear())
        
        # prediction parameters
        self.param('Decision threshold', decision_threshold = 0.5)
        self.param('Test on the training data', test_on_train = False)
        self.param('Plot intermediate behavior', plot = False)
    
    def evaluate(self,param):
        
        all_trial_files = list(set( [f for f in os.listdir(param.training_data_dir) if ".csv" in f] ))
        if param.training_strategy == 'within-subject':
            all_trials = list(set( [f for f in all_trial_files if param.subject_to_train_on in f] ))
        elif param.training_strategy == 'across-subject':
            all_trials = all_trial_files
        
        train_trials = np.random.choice(all_trials,size = int( len(all_trials)*param.train_test_split), replace=False ).flatten().tolist()
        test_trials = [e for e in all_trials if e not in train_trials]
        
        print('req. train trials: ', int( len(all_trials)*param.train_test_split))
        print('unique train trials: ', len(set(train_trials)))
        print('-----------------------------')
        print('act. test trials: ', len(test_trials))
        print('unique test trials: ', len(set(test_trials)))
        print('-----------------------------')
        print('all trials: ', len(all_trials))
        print('train+test trials: ', len(test_trials)+len(train_trials))
        print('train/test overlap: ', len(list(set(train_trials).intersection(test_trials))))
        print('all/test overlap: ', len(list(set(all_trials).intersection(test_trials))))
        
        train_df = pd.DataFrame()
        for train_trial in train_trials:
            temp_df = pd.read_csv(os.path.join(param.training_data_dir,train_trial),index_col=0)
            train_df = pd.concat([train_df,temp_df],axis=0,ignore_index=True)
        
        test_df = pd.DataFrame()        
        for test_trial in test_trials:
            temp_df = pd.read_csv(os.path.join(param.training_data_dir,test_trial),index_col=0)
            test_df = pd.concat([test_df,temp_df],axis=0,ignore_index=True)
        
        if param.included_features == 'All':
            train_xs = train_df[['AccX','AccY','AccZ','GyrX','GyrY','GyrZ','EulerX','EulerY','EulerZ']].values
            test_xs = test_df[['AccX','AccY','AccZ','GyrX','GyrY','GyrZ','EulerX','EulerY','EulerZ']].values
            size_in = 9         # 9 features of the fall detection data
        else:
            train_xs = train_df[param.included_features].values
            test_xs = test_df[param.included_features].values
            size_in = len(param.included_features)
        
        if param.normalization_method == 'Standard':
            StandardScaler(copy=False).fit_transform(train_xs )
            StandardScaler(copy=False).fit_transform(test_xs)
        elif param.normalization_method == 'Min/Max':
            MinMaxScaler(copy=False).fit_transform(train_xs)
            MinMaxScaler(copy=False).fit_transform(test_xs)
        
        lmu_train_xs = LDN(theta=param.theta, q=param.q, size_in=size_in).apply(train_xs)
        train_ys = train_df[['Fall/No Fall']].to_numpy() * param.scaling_factor
        test_ys = test_df[['Fall/No Fall']].to_numpy() * param.scaling_factor

        if param.test_on_train == True:
            # overwrite test data
            test_xs = train_xs
            test_ys = train_ys

        model = nengo.Network()
        with model:

            def stim_func(t):
                index = int(t/param.dt)-1
                return test_xs[index,:]
                
            stim = nengo.Node( size_out = size_in, output = stim_func )
            
            ldn = nengo.Node( LDN( theta = param.theta, q = param.q, size_in = size_in))
            nengo.Connection(stim, ldn, synapse = None)
            
            neurons = nengo.Ensemble(n_neurons = param.hl_neurons, dimensions = param.q*size_in, radius = param.hl_radius, neuron_type = param.hl_neuron_type)
            nengo.Connection(ldn, neurons, synapse = None)
            
            # initialize the network with the training data
            category = nengo.Ensemble(n_neurons = 100,dimensions = 1,radius = param.scaling_factor,neuron_type=nengo.RectifiedLinear())
            nengo.Connection(neurons, category, eval_points = lmu_train_xs, function = train_ys, synapse = None)
            
            p_stim = nengo.Probe(stim)
            p_ldn = nengo.Probe(ldn)
            p_neurons = nengo.Probe(neurons)
            p_category = nengo.Probe(category, synapse=0.01)

        sim = nengo.Simulator(model,dt=param.dt)
        with sim:
            sim.run(test_xs.shape[0]*param.dt)

        predictions = np.where(sim.data[p_category].flatten() > np.max(sim.data[p_category].flatten())*param.decision_threshold, 1, 0)    
        tn, fp, fn, tp = confusion_matrix( (test_ys/param.scaling_factor).astype(int), predictions ).ravel()
            
        for performance_metric, number in zip(('True Negatives','False Positives','False Negatives','True Positives'),(tn, fp, fn, tp)):
            print('{}: {}'.format(performance_metric,number))
            
        sensitivity = tp / (tp+fn)
        specificity = tn / (tn+fp)
        print('Sensitivity: {}%'.format(round(sensitivity*100,2)))
        print('Specificity: {}%'.format(round(specificity*100,2)))
            
        fig,(ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=(12,3))
        ax1.plot(sim.trange(), test_ys.flatten()/param.scaling_factor, label='True')
        #ax2.plot(sim.trange(), sim.data[p_neurons]/param.scaling_factor)
        
        ax2.plot(sim.trange(), sim.data[p_category]/param.scaling_factor)
        ax2.plot(sim.trange(), predictions, label='Predicted')

        ax1.legend()
        ax2.legend()
        plt.show()    
        
        scores = sim.data[p_category].flatten() / np.max(sim.data[p_category].flatten())
        neg_vals = scores < 0
        scores[neg_vals] = 0
        fpr,tpr,thresholds = roc_curve(y_true = (test_ys/param.scaling_factor).astype(int),y_score = scores)

        fig,ax = plt.subplots(1,1)
        ax.plot(fpr,tpr)
        plt.show()
        
        return {    
            'True negatives'    : tn,
            'False positives'   : fp,
            'False negatives'   : fn,
            'True positives'    : tp,
            'Sensitivity'       : sensitivity,
            'Specificity'       : specificity,
            }
        


