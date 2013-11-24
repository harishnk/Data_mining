import pandas as pd
import pylab as pl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
 
 
df = pd.read_csv("/users/harishnk/R/A2/BankDataModified.csv")
 
test_idx = np.random.uniform(0, 1, len(df)) <= 0.4
train = df[test_idx==True]
test = df[test_idx==False]
 
features = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Mortgage', 'Securities.Account', 'CD.Account', 'Online', 'CreditCard', 'dUndergrad', 'dGrad']
 
results = []

neighbors = 1
clf = KNeighborsClassifier(n_neighbors=neighbors)
clf.fit(train[features], train['Personal.Loan'])
preds = clf.predict(test[features])
accuracy = np.where(preds==test['Personal.Loan'], 1, 0).sum() / float(len(test))
print "Neighbors: %d, Accuracy: %3f" % (neighbors, accuracy)


 
