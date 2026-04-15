from data_prep import ingest_data
import matplotlib.pyplot as plt

X, _, _ = ingest_data('Project2Data.csv')

plt.boxplot(X, label = [s for s in X.columns], showfliers = False)
plt.show()