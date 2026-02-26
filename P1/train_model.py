import numpy as np
from pickle import load

with open('data.pkl', 'rb') as f:
    X, y = load(f)

print(X.shape, y.shape)