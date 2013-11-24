import pandas as pd
import pylab as pl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
 
 
df = pd.read_csv("/users/harishnk/R/A2/BankDataModified.csv")
 
#test_idx = np.random.uniform(0, 1, len(df)) <= 0.4
#train = df[test_idx==True]
#test = df[test_idx==False]

train = df[0:2999]
test = df[3000:4999]
 
features = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Mortgage', 'Securities.Account', 'CD.Account', 'Online', 'CreditCard', 'dUndergrad', 'dGrad']
 
results = []
for n in range(1, 50, 5):
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(train[features], train['Personal.Loan'])
    preds = clf.predict(test[features])
    accuracy = np.where(preds==test['Personal.Loan'], 1, 0).sum() / float(len(test))
    print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)
 
    results.append([n, accuracy])
 
results = pd.DataFrame(results, columns=["n", "accuracy"])

maxResults = []
print "Maximum accuracy is %3f" % (np.amax(results.accuracy, axis=0, out=maxResults))

 
pl.plot(results.n, results.accuracy)
pl.title("Accuracy with Increasing K")
pl.show()