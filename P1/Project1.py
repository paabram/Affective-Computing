from pathlib import Path
import numpy as np
import glob

def file_to_pair(file: Path) -> (str, ndarray):
    '''
        Reads single file into usable data for training
        file: Path to single file of space-separated facial data -> (y: emotion label (string), x: feature array (1, 249))
    '''
    # load csv and reshape to single row vector
    X = np.loadtxt(file, delimiter=" ")
    X = np.copy(X[:83, 1:]) # skip index, 83 rows
    X = X.reshape((1,-1)) # one row

    # get y as emotion from folder
    y = file.parts[-2]

    return (X, y)

path = Path('data')
X = np.ndarray()
y = []

_, y = file_to_row(Path('P1/data/F001/Angry/000.bnd'))
print(y)

# sklearn leaveoneout
# https://stackoverflow.com/questions/47204017/opening-and-reading-all-the-files-in-a-directory-in-python-python-beginner
