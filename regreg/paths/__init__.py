import numpy as np

from ..affine import astransform

def subsample_columns(X, columns):

    X = astransform(X)
    cols = np.zeros((len(columns),) + X.output_shape)
    indicator = np.zeros(X.input_shape[0]) # assuming 1-dimensional input and output shape here
    for i, col in enumerate(columns):
        indicator[col] = 1 # 1-hot vector
        cols[i] = X.dot(indicator)
        indicator[col] = 0 # back to 0 vector
    return cols.T
    

