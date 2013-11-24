import pandas as pd
import pylab as pl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
 
 
df = pd.read_csv("/users/harishnk/R/A2/BankDataModified.csv")
 
#test_idx = np.random.uniform(0, 1, len(df)) <= 0.4
#train = df[test_idx==True]
#test = df[test_idx==False]

train = df[0:2999]
test = df[3000:4999]

#columndelete = [0,1,2,3,4,5,6,7,8,10,11,12,13]
#test_result = np.delete(test, columndelete, axis=1)
 
features = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Mortgage', 'Securities.Account', 'CD.Account', 'Online', 'CreditCard', 'dUndergrad', 'dGrad']
 
results = []

neighbors = 6
clf = KNeighborsClassifier(n_neighbors=neighbors)
clf.fit(train[features], train['Personal.Loan'])
preds = clf.predict(test[features])
accuracy = np.where(preds==test['Personal.Loan'], 1, 0).sum() / float(len(test))
print "Neighbors: %d, Accuracy: %3f" % (neighbors, accuracy)

cm = confusion_matrix(test['Personal.Loan'], preds)

print cm


 
