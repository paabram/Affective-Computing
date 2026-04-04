import sys
from data_prep import ingest_data
from model_training import *

def perform_experiment(data_dir, signal):
    print(f'\nRunning experiment with {signal} data...')
    # load data
    X, y, subj_idx = ingest_data(data_dir, signal)
    # train models
    results = train_models(X, y, subj_idx)
    # get results
    render_results(results, signal, labels = y.unique())

# read command line arguments: data code in [1], data directory in [2]
if len(sys.argv) == 2: # if no transformation given
    for c in ['dia', 'sys', 'eda', 'res', 'all']:
        # perform all 5 experiments
        perform_experiment(sys.argv[1], c)
else:
    perform_experiment(sys.argv[2], sys.argv[1])