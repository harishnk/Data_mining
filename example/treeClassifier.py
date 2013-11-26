from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydot 

iris = load_iris()
clf = tree.DecisionTreeClassifier()

print iris.data[0:12]
print iris.target[0:12]

clf = clf.fit(iris.data, iris.target)

dot_data = StringIO.StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("iris.pdf") 