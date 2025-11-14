'''

14.06.2023

TO DO
- characterize inputs to determine needed dimensionality of LDN
- decorrelate input features and determine needed size_in

DOING
- LMU + 3-layer deep network integration (Iain)
- Fix training scheme / loading of training data (Collab? ComputeCanada? OpenCL? Sequential learning rule?)

DONE
- Fix discontinuities (Kathryn)
- Training across individuals (Kathryn)
- Performance quantification
- Turn this into a trial file to make it easier to play with parameters

BACKLOG
- Data augmentation (Kathryn)
- Selection of optimal hyperparams




13.06.2023
Today, we implemented an LMU classifier. 
We implemented a means of splitting the data set into test/train chunks

We learned that the dimensionality of the hidden layer/output layer has to be n*q
It consists of a 42-dimensional LDN w/ a 1000-neuron output layer from which we decode the output label. 
We find that the 'category' population can perform a reasonable approximation of the mapping of LDN-> labels
based on an experiment where we evaluated the network on the original training dataset.
The predictions

We do not know
- We suspect there's many 'discontinuities' just because of how the training/testing data are split
- These problems are compounded by the random initial seeds
	This is probably having a bigger effect on the train/test split

- variance of performance across individuals
#- Quantitatively, how well that function is approximated
- Quantitatively, the accuracy of the predictions
- Selection of optimal hyperparams

- We need to decide how to classify the prediction, as we're getting smooth values in the range [-100,100]
- Comparison to standard fall detection metrics
- Whether/how to integrate a deep network into the mix

'''