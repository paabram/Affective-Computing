import numpy as np
from pickle import load
from data_prep import *

# with open('data.pkl', 'rb') as f:
#     X, y = load(f)

X, y = load_data('data')

print(X.shape, y.shape)