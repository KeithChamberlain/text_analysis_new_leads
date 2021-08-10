import numpy as np
import pandas as pd
from sklearn import model_selection


'''
Train Test Split
'''
df = pd.read_csv("../data/data2protected.csv", compression="gzip", low_memory=False)
df_ = pd.read_csv("../data/data2confidential.csv", compression="gzip", low_memory=False)

y = df["nl_NewLead"]
y_ = df_["nl_NewLead"]

X = df
X_ = df_
X.drop(columns=["nl_NewLead"], inplace=True)
X_.drop(columns=["nl_NewLead"], inplace=True)


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20)
X__train, X__test, y__train, y__test = model_selection.train_test_split(X_, y_, test_size=0.20)



X_train.to_csv("../data/X_train_protected.csv", compression="gzip", index=False)
y_train.to_csv("../data/y_train_protected.csv", compression="gzip", index=False)
X_test.to_csv("../data/X_test_protected.csv", compression="gzip", index=False)
y_test.to_csv("../data/y_test_protected.csv", compression="gzip", index=False)

X__train.to_csv("../data/X__train_confidential.csv", compression="gzip", index=False)
y__train.to_csv("../data/y__train_confidential.csv", compression="gzip", index=False)
X__test.to_csv("../data/X__test_confidential.csv", compression="gzip", index=False)
y__test.to_csv("../data/y__test_confidential.csv", compression="gzip", index=False)

