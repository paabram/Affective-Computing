import sys
from data_prep import ingest_data

X, y, groups = ingest_data(sys.argv[2], sys.argv[1])
print(X)
print(y)
print(groups)