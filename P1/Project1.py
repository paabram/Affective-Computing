import sys
import numpy as np
from data_prep import load_data
from model_training import *

def perform_experiment(data_dir, trans_code):
    print(f'\nRunning experiment with code {trans_code}...')
    # load data
    X, y, subj_idx = load_data(data_dir, trans_code)
    # train models
    results = train_models(X, y, subj_idx)
    # get results
    render_results(results, trans_code, labels = np.unique(y))

# read command line arguments: transformation code in [1], data directory in [2]
if len(sys.argv) == 2: # if no transformation given
    for c in ['o', 't', 'x', 'y', 'z']:
        # perform all 5 experiments
        perform_experiment(sys.argv[1], c)
else:
    perform_experiment(sys.argv[2], sys.argv[1])