import sys
import numpy as np
import pandas as pd
import scipy
import matplotlib as plt
import sklearn
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
names = ['sepal_len','sepal_wid','petal_len','petal_wid','class']

#load dataset
mydata = pd.read_csv('iris.data.txt',names=names)

#summarize dataset
print(mydata.head(20))
print(mydata.shape)

#statistical summary
print(mydata.describe())

#class distribution
print(mydata.groupby('class').size())

#univariate plots
mydata.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)

#individual attribute histograms
mydata.hist()

#multivariate plots
scatter_matrix(mydata)

#copy dataset in array
array = mydata.values

#sep_h,sep_w,pet_h,pet_w in x
x = array[:,0:4]
#class in y
y = array[:,4]

#size of test set
validation_size=0.2

#splitting
x_train,x_validation,y_train,y_validation = model_selection.train_test_split(x,y,test_size=validation_size)

#different models testing
models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNA',KNeighborsClassifier()))
models.append(('DTC',DecisionTreeClassifier()))
models.append(('SVM',SVC()))

results=[]
names=[]

for name,model in models:
    #kfold method means split data into different parts and test them individually
    kfold = model_selection.KFold(n_splits=10,random_state=0.7)
    cv_results = model_selection.cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    names.append(name)
    msg = "%s:%f"%(name,cv_results.mean())
    print(msg)
    
#predicting with best model
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
predictions = knn.predict(x_validation)
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))