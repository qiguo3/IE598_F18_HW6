from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from pandas import Series
from pandas import DataFrame
from sklearn.model_selection import cross_val_score 

#load data
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

#Part 1: Random test train splits
list_randomstate=[]
list_accuracyscore_insample=[]
list_accuracyscore_outofsample=[]
for i in range (1,11):
    list_randomstate.append(i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i, stratify=y)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test) 
    tree = DecisionTreeClassifier(criterion='gini',max_depth=4, random_state=1)
    tree.fit(X_train_std, y_train) 
    y_pred = tree.predict(X_test_std)
    accuracy=accuracy_score(y_test,y_pred)
    y_pred_in=tree.predict(X_train_std)
    accuracy_in=accuracy_score(y_train,y_pred_in)
    list_accuracyscore_outofsample.append(accuracy)
    list_accuracyscore_insample.append(accuracy_in)

np_list_accuracyscore=np.vstack(list_accuracyscore_outofsample)
mean_out=np_list_accuracyscore.mean()
std_out=np_list_accuracyscore.std()
np_list_accuracyscore_insample=np.vstack(list_accuracyscore_insample)
mean_in=np_list_accuracyscore_insample.mean()
std_in=np_list_accuracyscore_insample.std()
np_accuracyscore=np.vstack((list_randomstate, list_accuracyscore_insample, list_accuracyscore_outofsample))   
df_accuracyscore=DataFrame(np_accuracyscore,columns=['','','','','','','','','',''])
df_accuracyscore.index=Series(['random_state','insample_accuracy_score','outofsample_accuracy_score'])
df_accuracyscore.to_excel('accuracyscore.xls')
print(df_accuracyscore)
print("mean_in",mean_in)
print("std_in",std_in)
print("mean_out",mean_out)
print("std_out",std_out)

#Part 2: Cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test) 
tree = DecisionTreeClassifier(criterion='gini',max_depth=4, random_state=1)
cvaccuracyscore=cross_val_score(tree, X_train_std, y_train, cv=10)
cv_mean=cvaccuracyscore.mean()
cv_std=cvaccuracyscore.std()
print(cvaccuracyscore)
print("CV mean",cv_mean)
print("CV std", cv_std)
cvaccuracyscore=cross_val_score(tree, X_train_std, y_train, cv=10)
tree.fit(X_train_std, y_train)
y_pred=tree.predict(X_test_std)
testaccuracy=accuracy_score(y_test,y_pred)
print("test accuracy", testaccuracy)

print("My name is QI GUO")
print("My NetID is: qiguo3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################