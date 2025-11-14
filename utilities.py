import pandas as pd
import numpy as np

def sample_with_minimum_distance(n=40, k=4, d=10):
    """
    Sample of k elements from range(n), with a minimum distance d.
    """
    sample = random.sample(range(n-(k-1)*(d-1)), k)
    return [s + (d-1)*r for s, r in zip(sample, ranks(sample))]


def generate_train_test_split(ddf, chunk_size = 200):
    print(ddf.head())
    
    # generate random indices
    
    # take a chunk
    # copy to a new dataframe
    
    
    
    
    return 'none'