from pathlib import Path
import numpy as np
from glob import glob
from pickle import dump

def file_to_vec(file: Path) -> (str, np.ndarray):
    '''
        Reads single file into usable data for training
        file: Path to single file of space-separated facial data -> (y: emotion label (string), x: feature array (1, 249))
    '''
    # load csv and reshape to single row vector
    X = np.loadtxt(file, delimiter=" ")
    X = np.copy(X[:83, 1:]) # skip index, 83 rows
    X = X.reshape((1,-1)) # one row

    # get y as emotion from folder and participant index
    y = file.parts[-2]
    subj = file.parts[-3]

    return (X, y, subj)

def load_data(file: str) -> (np.ndarray, np.ndarray, list):
    '''Given a directory path, load all data into X and y arrays, as well as matching list of subject indices for cross-validation'''
    path = Path(file)
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
                # append data to X and label to y
                X.append(x)
                y.append(label)
                subj_idx.append(subj)
    
    X = np.array(X).squeeze()
    y = np.array(y)
    subj_idx = np.array(subj_idx)

    return (X, y, subj_idx)

if __name__ == '__main__':
    X, y, subj_idx = load_data('P1/data/')
    print(X.shape, y.shape, len(subj_idx))
    # pickle to speed up recall during testing
    # with open('data.pkl', 'wb') as f:
    #     dump((X, y), f)

# sklearn leaveoneout
# https://stackoverflow.com/questions/47204017/opening-and-reading-all-the-files-in-a-directory-in-python-python-beginner
