import pandas as pd

def read_csv_var_length(path: str) -> pd.DataFrame:
    '''
        Read in csv where each row contains a variable number of datapoint columns, using numeric column names.
            path: csv file containing physiological data in the assignment format: subject id, measured signal, pain label, data point columns
        -> df: pandas DataFrame with data in path, having subj_id, signal, label, and numbered data columns starting at 3
    '''
    max_cols = 0
    # count the data lines to get max length
    with open(path, 'r') as temp_f:
        # read file
        lines = temp_f.readlines()
        # iterate lines
        for l in lines:
            # count the column length for the current line
            column_count = len(l.split(',')) + 1
            # update max
            max_cols = max(column_count, max_cols)

    # use max_cols to generate column names (0, 1, 2, ..., max - 1)
    column_names = [i for i in range(0, max_cols)]
    # read csv, passing in column_names to read all rows to max length
    df = pd.read_csv(path, header=None, names=column_names)
    # rename first 3 columns
    df = df.rename(columns={0:'subj_id', 1:'signal', 2:'label'})

    return df

def transform_data(df: pd.DataFrame) -> pd.Dataframe:
    '''
        Given DataFrame in wide format (long rows of data points), melt and aggregate to 16 columns in assignment (signal x [mean, var, min, max]).
            df: wide DataFrame as returned by read_csv_var_length()
        -> df_agg: pivot table-like DataFrame aggregating df datapoints into format usable for training
    '''
    # melt numbered columns long so each set of (subject, label, signal) occurs in (n timesteps) rows
    df_long = df.melt(id_vars=['subj_id', 'label', 'signal'], var_name='timestep')
    # aggregate values with pivot_table: subject and label on rows, and signal type on columns
    df_agg = df_long.pivot_table('value', index=['subj_id', 'label'], columns='signal', aggfunc=['mean', 'var', 'max', 'min'])
    # flatten multi-index to usable names
    df_agg.columns = [f"{agg}_{signal}" for agg, signal in df_agg.columns]
    # move subj_id, label from index back into columns
    df_agg = df_agg.reset_index()

    return df_agg

def ingest_data(path: str, signal: str = 'all') -> (pd.DataFrame, pd.Series, pd.Series):
    '''
        Given path to data file and experiment type, return y series (pain label), X table (columns for desired experiment), and subject group (corresponding subj_id Series, used for cv grouping).
            path: csv file containing physiological data in the assignment format
            signal: indicates what signal to use for model, one of: dia, sys, eda, res, all (default)
        -> X: DataFrame of desired columns
        -> y: Series of pain labels
        -> subjs: subj_id Series aligned with y and X to be used for cross validation
    '''
    # read data
    df = read_csv_var_length(path)
    # aggregate data points
    df = transform_data(df)
    # extract y
    y = df['label']
    # get groups
    subjs = df['subj_id']
    # get X columns and filter by desired pattern
    X = df.drop(columns=['label', 'subj_id'])
    if signal != 'all':
        X = X.loc[:, X.columns.str.contains(signal, case=False, regex=False)]

    return X, y, subjs
