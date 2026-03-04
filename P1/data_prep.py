from pathlib import Path
import numpy as np
from glob import glob
from math import pi, cos, sin

def file_to_vec(file: Path) -> (np.ndarray, str, str):
    '''
        Reads single file into X table, y label, and subject ID
            file: Path to single file of space-separated facial data 
        -> X: feature table (83, 3), y: emotion label (string), subj: subject ID (string)
    '''
    # load csv and ensure correct dimensions
    X = np.loadtxt(file, delimiter=" ")
    X = np.copy(X[:83, 1:]) # skip index, 83 rows

    # get y as emotion from folder and participant index
    y = file.parts[-2]
    subj = file.parts[-3]

    return (X, y, subj)

def transform_X(X: np.ndarray, trans_code: str):
    '''
        Transform table of facial landmarks according to given single-character code (per assignment)
            X: (3, 84) array of 3D coordinates x, y, z
            code: one of [o, t, x, y, z], indicating data transform to do
        -> single row vector of transformed X data with (1, 249)
    '''
    # unmodified
    if trans_code == 'o':
        return X.reshape((1,-1))

    # translated to center
    if trans_code == 't':
        X_new = X - X.mean(axis = 0) # subtract column-wise mean from each point
        return X_new.reshape((1,-1))

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
    return X_new.reshape((1,-1))

def load_data(dir: str, trans_code: str = 'o') -> (np.ndarray, np.ndarray, list):
    '''
        Given a directory path, load all data into X and y arrays, as well as matching list of subject indices for cross-validation. Transform as desired
            dir: name of directory to process
            trans_code: see transform_X
        -> X: array with (n, 249), where n is # data files across all subjects, y: n corresponding emotion labels, subj_idx: n corresponding subject IDs
    '''
    path = Path(dir)
    X = []
    y = []
    subj_idx = []

    # outer loop: subjects
    for subject in path.iterdir():
        if not subject.is_dir(): 
                continue
        # inner loop: emotions
        for emotion in subject.iterdir():
            if not emotion.is_dir(): 
                continue
            # innermost loop: files
            files = emotion.glob('*.bnd')
            for f in files:
                x, label, subj = file_to_vec(f)
                x = transform_X(x, trans_code)
                # append data to X and label to y
                X.append(x)
                y.append(label)
                subj_idx.append(subj)
    
    # convert all to np arrays for speed
    X = np.array(X).squeeze() # remove extra dimension
    y = np.array(y)
    subj_idx = np.array(subj_idx)

    return (X, y, subj_idx)