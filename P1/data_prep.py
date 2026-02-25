from pathlib import Path
import numpy as np
from glob import glob
from pickle import dump

def file_to_pair(file: Path) -> (str, np.ndarray):
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

# --- READ DATA ---

path = Path('data/')
X = []
y = []

# outer loop: subjects
for subject in path.iterdir():
    # inner loop: emotions
    for emotion in subject.iterdir():
        # innermost loop: files
        files = emotion.glob('*.bnd')
        for f in files:
            x, label = file_to_pair(f)
            # append data to X and label to y
            X.append(x)
            y.append(label)
 
X = np.array(X).squeeze()
y = np.array(y)

print(X.shape, y.shape)
# pickle to speed up recall later
with open('data.pkl', 'wb') as f:
    dump((X, y), f)

# sklearn leaveoneout
# https://stackoverflow.com/questions/47204017/opening-and-reading-all-the-files-in-a-directory-in-python-python-beginner
