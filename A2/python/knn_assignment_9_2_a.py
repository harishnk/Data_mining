import pandas as pd
import pylab as pl
import numpy as np
import pydot
from sklearn import tree
from sklearn.externals.six import StringIO
import sys
from inspect import getmembers
 
download = ''
df = pd.read_csv("/users/harishnk/datamining/A2/FlightDelays.csv")
 
#test_idx = np.random.uniform(0, 1, len(df)) <= 0.4
#train = df[test_idx==True]
#test = df[test_idx==False]

#train = df[0:2999]
#test = df[3000:4999]

df['Flight Status'].replace(to_replace=['ontime', 'delayed'], value=[0,1], inplace=True)

#print df['Flight Status'][1:50]

 
#features = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Mortgage', 'Securities.Account', 'CD.Account', 'Online', 'CreditCard', 'dUndergrad', 'dGrad']
#features = ['CRS_DEP_TIME', 'CARRIER', 'DEP_TIME', 'DEST', 'DISTANCE', 'FL_DATE', 'FL_NUM', 'ORIGIN', 'Weather', 'DAY_WEEK', 'DAY_OF_MONTH', 'TAIL_NUM']
outcome = ['Flight Status']
#excludes field 'DEP_TIME', the actual departure time as it cannot be used as predictor
#data = ['CRS_DEP_TIME', 'CARRIER', 'DEST', 'DISTANCE', 'FL_DATE', 'FL_NUM', 'ORIGIN', 'Weather', 'DAY_WEEK', 'DAY_OF_MONTH', 'TAIL_NUM', 'Flight Status']
data = ['DISTANCE', 'Weather', 'DAY_OF_MONTH', 'Flight Status']
#dummies = ['DAY_WEEK', 'CARRIER', 'ORIGIN', 'DEST']
#, 'CARRIER', 'ORIGIN', 'DEST'

dummies_day_week = pd.get_dummies(df['DAY_WEEK'], prefix='d1')

dummies_carrier = pd.get_dummies(df['CARRIER'], prefix='d2')

dummies_origin = pd.get_dummies(df['ORIGIN'], prefix='d3')

dummies_destination = pd.get_dummies(df['DEST'], prefix='d4')

df_withDummiesNoBins = df[data].join(dummies_day_week).join(dummies_carrier).join(dummies_origin).join(dummies_destination)

#df_d2 = df_d1.join(dummies_carrier)

#print df_withDummiesNoBins[1:5]

bins = [600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

dummiesAndBins = pd.get_dummies(pd.cut(df['CRS_DEP_TIME'], bins))

df_withDummiesAndBins = df_withDummiesNoBins.join(dummiesAndBins)

if download=='X':
  df_withDummiesAndBins.to_csv("/users/harishnk/datamining/A2/FlightDelaysTransformed.csv")


train = df_withDummiesAndBins[0:1201]
validate = df_withDummiesAndBins[1202:2201]

DummiesAndBinsHeader = list(train.columns.values) #.remove('Flight Status')
DummiesAndBinsHeader.remove('Flight Status')

DummiesAndBinsHeader = list(validate.columns.values) #.remove('Flight Status')
DummiesAndBinsHeader.remove('Flight Status')
#print DummiesAndBinsHeader

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6)
clf = clf.fit(train[DummiesAndBinsHeader], train['Flight Status'])

clf.predict(validate[DummiesAndBinsHeader])

#print( getmembers( clf.tree_.threshold ) )

dot_data = StringIO()
if sys.getsizeof(dot_data) > 0:
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("/users/harishnk/datamining/A2/flightdelay.pdf")

 
# results = []
# for n in range(1, 50, 5):
#     clf = KNeighborsClassifier(n_neighbors=n)
#     clf.fit(train[features], train['Personal.Loan'])
#     preds = clf.predict(test[features])
#     accuracy = np.where(preds==test['Personal.Loan'], 1, 0).sum() / float(len(test))
#     print "Neighbors: %d, Accuracy: %3f" % (n, accuracy)
 
#     results.append([n, accuracy])
 
# results = pd.DataFrame(results, columns=["n", "accuracy"])

# maxResults = []
# print "Maximum accuracy is %3f" % (np.amax(results.accuracy, axis=0, out=maxResults))

 
# pl.plot(results.n, results.accuracy)
# pl.title("Accuracy with Increasing K")
# pl.show()

#preds = gnb.fit(train[features], train['Personal.Loan']).predict(train[features])

#testPoint = [1,1]

#gnb.predict(testPoint)
#print gnb.predict_proba(testPoint)

##print 'accuracy is =', gnb.score(train[features],train['Personal.Loan'])

##print("Number of miscategorized points: %d" % (train[features] != preds).sum())

#mnb = naive_bayes.MultinomialNB()
#mnb.fit(train[features], train['Personal.Loan'])

#print "Classification accuracy of MNB = ", mnb.score(test[features], test['Personal.Loan'])

