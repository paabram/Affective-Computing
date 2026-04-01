import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi

# copied from data_prep.py, but need whole table not single row vector
def transform_X(X: np.ndarray, trans_code: str):
    # unmodified
    if trans_code == 'o':
        return X

    # translated to center
    if trans_code == 't':
        X_new = X - X.mean(axis = 0) # subtract column-wise mean from each point
        return X_new

    # mirrored x-axis
    if trans_code == 'x':
        scale_mat =  np.array([[1, 0, 0], \
                               [0, cos(pi), sin(pi)], \
                               [0, (-1)*sin(pi), cos(pi)]])
    # mirrored y-axis
    elif trans_code == 'y':
        scale_mat =  np.array([[cos(pi), 0, (-1)*sin(pi)], \
                               [0, 1, 0], \
                               [sin(pi), 0, cos(pi)]])
    # mirrored z-axis
    elif trans_code == 'z':
        scale_mat =  np.array([[cos(pi), sin(pi), 0], \
                               [(-1)*sin(pi), cos(pi), 0], \
                               [0, 0, 1]])                        
    
    X_new = X @ scale_mat.T # equivalent transform to scale_mat @ X[i] but all rows at once
    return X_new

file = 'data/F001/Angry/000.bnd'
X = np.loadtxt(file, delimiter=' ')
X = np.copy(X[:83, 1:]) # skip index, 83 rows

for c in ['o', 't', 'x', 'y', 'z']:
    X_new = transform_X(X, c)
    # 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X_new[:,0], X_new[:,1], X_new[:,2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()