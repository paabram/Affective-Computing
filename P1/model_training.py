import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from time import localtime, strftime
import json

def train_models(X: np.ndarray, y: np.ndarray, subj_idx: np.ndarray) -> dict:
    '''
        Given prepared X, y, and group vectors, train k models with LOSO CV. Return average accuracy, precision, recall, and confusion matrix
            X: (n, 249) array of data across all subjects
            y: corresponding emotion labels
            groups: corresponding subject IDs
        -> dictionary with averaged accuracy, precision, recall, and confusion matrix across all k folds
    '''
    # initialize
    nclasses = len(np.unique(y))
    n_subjs = 0
    acc_sum = 0
    prec_sum = 0
    rec_sum = 0
    confusion = np.zeros((nclasses, nclasses))

    # one subject's data (as identified by index of subjects to rows) will be left out for testing each model
    loso = LeaveOneGroupOut()
    for train, test in loso.split(X, y, groups = subj_idx):
        # progress bar
        if n_subjs % 5 == 0:
            print(f'Computing LOSO Fold {n_subjs}...')

        # split
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        
        # train and predict
        # why random forest: minimal overfitting, nonlinear, robust to outliers, faster and easier than svm
        model = RandomForestClassifier(random_state=67).fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # running sum for average scores
        acc_sum += accuracy_score(y_test, y_pred)
        prec_sum += precision_score(y_test, y_pred, average='micro')
        rec_sum += recall_score(y_test, y_pred, average='micro')
        confusion += confusion_matrix(y_test, y_pred)
        n_subjs += 1
    
    # average across all subjects
    results = {'accuracy': acc_sum/n_subjs, \
               'precision': prec_sum/n_subjs, \
               'recall':rec_sum/n_subjs, \
               'confusion_mat': confusion/n_subjs}

    return results

def render_results(results: dict, file_suffix: str, labels: np.ndarray):
    '''
        Display results of model training  and save to JSON, plus confusion matrix graphic to jpeg file. Identify by suffix, which will be the transform code passed in, and timestamp
            results: dictionary as returned by train_models
            file_suffix: transformation code to identify experiments
            labels: emotion labels for confusion matrix
    '''
    timestamp = strftime('%m-%d-%H:%M', localtime())
    # print results
    print('-' * 64)
    header = f'Results of experiment with transform {file_suffix} ({timestamp})'
    print(f'{header:^64}\n')
    print(f'Accuracy: {results['accuracy']}')
    print(f'Precision: {results['precision']}')
    print(f'Recall: {results['recall']}')

    # confusion matrix print out
    confusion = results['confusion_mat']
    print(f'\nConfusion Matrix (True ~ Predicted, labels = {[str(l) for l in labels]})')
    for r in confusion:
        for i in r:
            print(f'[{i:^7.2f}]', end = '')
        print()
        
    # render confusion matrix
    disp = ConfusionMatrixDisplay(confusion, display_labels=labels)
    disp.plot()
    plt.savefig(f'results/confusionMatrix_{file_suffix}_{timestamp}.jpg') # save to file
    plt.show()

    # write json results
    results['confusion_mat'] = confusion.tolist()
    with open(f'results/results_{file_suffix}_{timestamp}.txt', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent = 4)