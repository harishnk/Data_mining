import pandas as pd
import pylab as pl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
 
 
df = pd.read_csv("/users/harishnk/datamining/A2/BankDataModified.csv")
 
#test_idx = np.random.uniform(0, 1, len(df)) <= 0.4
#train = df[test_idx==True]
#test = df[test_idx==False]

train = df[0:2499]
validate = df[2500:3999]
test = df[4000:4999]

#columndelete = [0,1,2,3,4,5,6,7,8,10,11,12,13]
#test_result = np.delete(test, columndelete, axis=1)
 
features = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Mortgage', 'Securities.Account', 'CD.Account', 'Online', 'CreditCard', 'dUndergrad', 'dGrad']
 


neighbors = 6
clf = KNeighborsClassifier(n_neighbors=neighbors)
clf.fit(train[features], train['Personal.Loan'])

predsTrain = clf.predict(train[features])
accuracy = np.where(predsTrain==train['Personal.Loan'], 1, 0).sum() / float(len(train))
print "Train: Neighbors: %d, Accuracy: %3f" % (neighbors, accuracy)
cm = confusion_matrix(train['Personal.Loan'], predsTrain)
print cm

predsValidate = clf.predict(validate[features])
accuracy = np.where(predsValidate==validate['Personal.Loan'], 1, 0).sum() / float(len(validate))
print "Validate: Neighbors: %d, Accuracy: %3f" % (neighbors, accuracy)
cm = confusion_matrix(validate['Personal.Loan'], predsValidate)
print cm


predsTest = clf.predict(test[features])
accuracyTest = np.where(predsTest==test['Personal.Loan'], 1, 0).sum() / float(len(test))
print "Test: Neighbors: %d, Accuracy: %3f" % (neighbors, accuracyTest)
cmTest = confusion_matrix(test['Personal.Loan'], predsTest)
print cmTest



 
