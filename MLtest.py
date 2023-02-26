import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import sklearn
from sklearn import neighbors, naive_bayes, svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
def load_dataset(file_name, target_column):
    df=pd.read_csv(file_name)
    TRAINING_FEATURES = df.columns[(df.columns != target_column) &
                                   (df.columns!='Author')&
                                   (df.columns!='rating')&
                                   (df.columns!='rating_count')&
                                   (df.columns!='Ganer')&
                                   (df.columns!='book_name')]

    X = df[TRAINING_FEATURES]
    y = df[target_column]
    return X, y

df='finaly result logestic.csv'
X,y=load_dataset(df, 'is_successful')
#split the data to train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#logistic regration model

#fit(train)
def train_by_logistic(X_train, y_train):
    trained_LogisticRegression=LogisticRegression(max_iter=10000,solver="lbfgs")
    trained_LogisticRegression.fit(X_train, y_train)
    return trained_LogisticRegression

logistic_model=train_by_logistic(X_train, y_train)

#predict
def predict_by_logistic(trained_2nd_model, X_test):
    predicted_vals=trained_2nd_model.predict(X_test)
    return predicted_vals

predict_model=predict_by_logistic(logistic_model, X_test)

#evaluate
def evaluate_performance_by_logistic(y_test,y_predicted):
    evaluate_value=f1_score(y_test, y_predicted)
    return evaluate_value

print(evaluate_performance_by_logistic(y_test,predict_model))
print('------------------')
print('This is how accurate the predictions are:\n' + (str)((int)((evaluate_performance_by_logistic(y_test,predict_model))*100)) + '% on ' + (str)((int)(len(y)*0.5)) + ' data')



def find_best_random_forest_num_estimators(X_train, y_train):
    parameters = {'n_estimators':[15,10] }
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, parameters,scoring=make_scorer(metrics.f1_score, greater_is_better=True))
    #fit
    clf.fit(X_train, y_train)
    return clf

#predict
Random_Forest=find_best_random_forest_num_estimators(X_train, y_train)

print(f1_score(y_test,Random_Forest.predict(X_test)))



trained_LogisticRegression=LogisticRegression(max_iter=10000,solver="lbfgs")
rf = RandomForestClassifier(n_estimators=50, random_state=1)
clf=svm.SVC(probability=True)
est_Ensemble = VotingClassifier(estimators=[('Logistic Regression', trained_LogisticRegression), ('Random Forest', rf)],voting='soft',weights=[1,1])
est_Ensemble .fit(X_train, y_train)
y_predicted=est_Ensemble.predict(X_test)
print(f1_score(y_test, y_predicted))