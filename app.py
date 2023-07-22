import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from google.colab import drive
drive.mount('/content/drive/')

df= pd.read_csv("/content/drive/MyDrive/heart.csv")

df= df.drop(['oldpeak','slp','thall'],axis=1)

df.head()

df.shape

df.isnull().sum()

df.corr()

sns.heatmap(df.corr())

plt.figure(figsize=(20, 10))
plt.title("Age of Patients")
plt.xlabel("Age")
sns.countplot(x='age',data=df)

plt.figure(figsize=(20, 10))
plt.title("Sex of Patients,0=Female and 1=Male")

sns.countplot(x='sex',data=df)

cp_data= df['cp'].value_counts().reset_index()
cp_data['index'][3]= 'asymptomatic'
cp_data['index'][2]= 'non-anginal'
cp_data['index'][1]= 'Atyppical Anigma'
cp_data['index'][0]= 'Typical Anigma'
cp_data

plt.figure(figsize=(20, 10))
plt.title("Chest Pain of Patients")

sns.barplot(x=cp_data['index'],y= cp_data['cp'])

ecg_data= df['restecg'].value_counts().reset_index()
ecg_data['index'][0]= 'normal'
ecg_data['index'][1]= 'having ST-T wave abnormality'
ecg_data['index'][2]= 'showing probable or definite left ventricular hypertrophy by Estes'

ecg_data

plt.figure(figsize=(20, 10))
plt.title("ECG data of Patients")

sns.barplot(x=ecg_data['index'],y= ecg_data['restecg'])

"""#### This is our ECG Data"""

sns.pairplot(df,hue='output',data=df)

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.distplot(df['trtbps'], kde=True, color = 'magenta')
plt.xlabel("Resting Blood Pressure (mmHg)")
plt.subplot(1,2,2)
sns.distplot(df['thalachh'], kde=True, color = 'teal')
plt.xlabel("Maximum Heart Rate Achieved (bpm)")

plt.figure(figsize=(10,10))
sns.distplot(df['chol'], kde=True, color = 'red')
plt.xlabel("Cholestrol")

df.head()

from sklearn.preprocessing import StandardScaler

scale=StandardScaler()

scale.fit(df)

df= scale.transform(df)

df=pd.DataFrame(df,columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
       'exng', 'caa', 'output'])

df.head()

x= df.iloc[:,:-1]
x

y= df.iloc[:,-1:]
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

lbl= LabelEncoder()

encoded_y= lbl.fit_transform(y_train)

logreg= LogisticRegression()

logreg = LogisticRegression()
logreg.fit(x_train, encoded_y)

Y_pred1

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

encoded_ytest= lbl.fit_transform(y_test)

Y_pred1 = logreg.predict(x_test)
lr_conf_matrix = confusion_matrix(encoded_ytest,Y_pred1 )
lr_acc_score = accuracy_score(encoded_ytest, Y_pred1)

lr_conf_matrix

print(lr_acc_score*100,"%")

from sklearn.tree import DecisionTreeClassifier

tree= DecisionTreeClassifier()

tree.fit(x_train,encoded_y)

ypred2=tree.predict(x_test)

encoded_ytest= lbl.fit_transform(y_test)

tree_conf_matrix = confusion_matrix(encoded_ytest,ypred2 )
tree_acc_score = accuracy_score(encoded_ytest, ypred2)

tree_conf_matrix

print(tree_acc_score*100,"%")

from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier()

rf.fit(x_train,encoded_y)

ypred3 = rf.predict(x_test)

rf_conf_matrix = confusion_matrix(encoded_ytest,ypred3 )
rf_acc_score = accuracy_score(encoded_ytest, ypred3)

rf_conf_matrix

print(rf_acc_score*100,"%")

from sklearn.neighbors import KNeighborsClassifier

error_rate= []
for i in range(1,40):
    knn= KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,encoded_y)
    pred= knn.predict(x_test)
    error_rate.append(np.mean(pred != encoded_ytest))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.xlabel('K Vlaue')
plt.ylabel('Error rate')
plt.title('To check the correct value of k')
plt.show()

knn= KNeighborsClassifier(n_neighbors=12)
knn.fit(x_train,encoded_y)
ypred4= knn.predict(x_test)

knn_conf_matrix = confusion_matrix(encoded_ytest,ypred4 )
knn_acc_score = accuracy_score(encoded_ytest, ypred4)

knn_conf_matrix

print(knn_acc_score*100,"%")

from sklearn import svm

svm= svm.SVC()

svm.fit(x_train,encoded_y)

ypred5= svm.predict(x_test)

svm_conf_matrix = confusion_matrix(encoded_ytest,ypred5)
svm_acc_score = accuracy_score(encoded_ytest, ypred5)

svm_conf_matrix

print(svm_acc_score*100,"%")

model_acc= pd.DataFrame({'Model' : ['Logistic Regression','Decision Tree','Random Forest','K Nearest Neighbor','SVM'],'Accuracy' : [lr_acc_score*100,tree_acc_score*100,rf_acc_score*100,knn_acc_score*100,svm_acc_score*100]})

model_acc = model_acc.sort_values(by=['Accuracy'],ascending=False)

model_acc

from sklearn.ensemble import AdaBoostClassifier

adab= AdaBoostClassifier(base_estimator=svm,n_estimators=100,algorithm='SAMME',learning_rate=0.01,random_state=0)

adab.fit(x_train,encoded_y)

ypred6=adab.predict(x_test)

adab_conf_matrix = confusion_matrix(encoded_ytest,ypred6)
adab_acc_score = accuracy_score(encoded_ytest, ypred6)

adab_conf_matrix

print(adab_acc_score*100,"%")

adab.score(x_train,encoded_y)

adab.score(x_test,encoded_ytest)

"""#### As we see our model has performed very poorly with just 50% accuracy

#### We will use Grid Seach CV for HyperParameter Tuning
"""

from sklearn.model_selection import GridSearchCV

model_acc

param_grid= {

    'solver': ['newton-cg', 'lbfgs', 'liblinear','sag', 'saga'],
    'penalty' : ['none', 'l1', 'l2', 'elasticnet'],
    'C' : [100, 10, 1.0, 0.1, 0.01]

}

grid1= GridSearchCV(LogisticRegression(),param_grid)

grid1.fit(x_train,encoded_y)

grid1.best_params_

logreg1= LogisticRegression(C=0.01,penalty='l2',solver='liblinear')

logreg1.fit(x_train,encoded_y)

logreg_pred= logreg1.predict(x_test)

logreg_pred_conf_matrix = confusion_matrix(encoded_ytest,logreg_pred)
logreg_pred_acc_score = accuracy_score(encoded_ytest, logreg_pred)

logreg_pred_conf_matrix

print(logreg_pred_acc_score*100,"%")

n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']

grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)

from sklearn.model_selection import RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=knn, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

grid_search.fit(x_train,encoded_y)

grid_search.best_params_

knn= KNeighborsClassifier(n_neighbors=12,metric='manhattan',weights='distance')
knn.fit(x_train,encoded_y)
knn_pred= knn.predict(x_test)

knn_pred_conf_matrix = confusion_matrix(encoded_ytest,knn_pred)
knn_pred_acc_score = accuracy_score(encoded_ytest, knn_pred)

knn_pred_conf_matrix

print(knn_pred_acc_score*100,"%")

kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']

grid = dict(kernel=kernel,C=C,gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=svm, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

grid_search.fit(x_train,encoded_y)

grid_search.best_params_

from sklearn.svm import SVC

svc= SVC(C= 0.1, gamma= 'scale',kernel= 'sigmoid')

svc.fit(x_train,encoded_y)

svm_pred= svc.predict(x_test)

svm_pred_conf_matrix = confusion_matrix(encoded_ytest,svm_pred)
svm_pred_acc_score = accuracy_score(encoded_ytest, svm_pred)

svm_pred_conf_matrix

print(svm_pred_acc_score*100,"%")

logreg= LogisticRegression()
logreg = LogisticRegression()
logreg.fit(x_train, encoded_y)

Y_pred1

lr_conf_matrix

print(lr_acc_score*100,"%")

# Confusion Matrix of  Model enlarged
options = ["Disease", 'No Disease']

fig, ax = plt.subplots()
im = ax.imshow(lr_conf_matrix, cmap= 'Set3', interpolation='nearest')

# We want to show all ticks...
ax.set_xticks(np.arange(len(options)))
ax.set_yticks(np.arange(len(options)))
# ... and label them with the respective list entries
ax.set_xticklabels(options)
ax.set_yticklabels(options)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(options)):
    for j in range(len(options)):
        text = ax.text(j, i, lr_conf_matrix[i, j],
                       ha="center", va="center", color="black")

ax.set_title("Confusion Matrix of Logistic Regression Model")
fig.tight_layout()
plt.xlabel('Model Prediction')
plt.ylabel('Actual Result')
plt.show()
print("ACCURACY of our model is ",lr_acc_score*100,"%")

import pickle

pickle.dump(logreg,open('heart.pkl','wb'))

!pip install evalml

df= pd.read_csv("/content/drive/MyDrive/heart.csv")

df.head()

"""Let us split our Data Set into Dependent i.e our Targer variable and independent variable"""

x= df.iloc[:,:-1]
x

y= df.iloc[:,-1:]
y= lbl.fit_transform(y)
y

import evalml

"""Eval ML Library will do all the pre processing techniques for us and split the data for us"""

X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(x, y, problem_type='binary')

"""There are different problem type parameters in Eval ML, we have a Binary type problem here, that's why we are using Binary as a input"""

evalml.problem_types.ProblemTypes.all_problem_types

from evalml.automl import AutoMLSearch
automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='binary')
automl.search()

"""As we see from the above output thge Auto ML Classifier has given us the best fit Algorithm which is Extra Trees Classifier with Imputer
We can also commpare the rest of the models
"""

automl.rankings

automl.best_pipeline

best_pipeline=automl.best_pipeline

"""We can have a Detailed description of our Best Selected Model"""

automl.describe_pipeline(automl.rankings.iloc[0]["id"])

best_pipeline.score(X_test, y_test, objectives=["auc","f1","Precision","Recall"])

"""Now if we want to build our Model for a specific objective we can do that"""

automl_auc = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='binary', objective='auc', additional_objectives=['f1', 'precision'], max_batches=1, optimize_thresholds=True)

automl_auc.search()

automl_auc.rankings

automl_auc.describe_pipeline(automl_auc.rankings.iloc[0]["id"])

best_pipeline_auc = automl_auc.best_pipeline

# get the score on holdout data
best_pipeline_auc.score(X_test, y_test,  objectives=["auc"])

"""We got an 88.5 % AUC Score which is the highest of all

Save the model
"""

best_pipeline.save("model.pkl")

"""Loading our Model"""

final_model=automl.load('model.pkl')

final_model.predict_proba(X_test)

