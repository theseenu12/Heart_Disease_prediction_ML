import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV,cross_val_score
from sklearn.linear_model import LinearRegression,LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,f1_score,accuracy_score,precision_score,confusion_matrix,ConfusionMatrixDisplay,roc_curve,RocCurveDisplay
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from joblib import load,dump

## getting all the optimizers,lossfunctions and activation functions below by using keras
# from tensorflow import keras
# from keras.optimizers import 
# from keras.activations import 
# from keras.losses import 

## Predicting Heart Disease Using Machine Learning

## data 

# the original data came from UCI Machine Learning repository 
# There is also version available in the kaggle

## preparing the tools
## we are going to us pandas,numpy,matplotlib or seaborn for data analysis


## load the data

df = pd.read_csv('heart_dataset.csv')

# print(df)

# print(df.iloc[1:10,1:5])

## data exploration
# the goal here is to find about more about the data and become subject matter expert on the dataset youre woking with

## 1 what questions are you trying to solve
## 2 what kind of data do you have and how do we treat different types such as numberical or categorical
## 3 whats missing from the data and how do you deal with it
## 4 where are the outliers and how do you deal with it
## 5 how can you add change and remove feautures to get more of your data

# print(df.tail(10))

print(df['target'].value_counts())

# df['target'].value_counts().plot.bar()

# plt.show()

print(df.info())

## check if there are any missing values

print(df.isna().sum())

## check all the mean and statistical information of the data using pd.describe()

print(df.describe())

print(df['target'].describe())

## check the correlation of each column in the dataset

print(df.corr())

print(df.corrwith(df['target']))


## heart disease frequency according to sex

print(df['sex'].value_counts())

## compare target column with sex column
print(pd.crosstab(df['target'],df['sex']))

## create a plot of crosstab
# pd.crosstab(df['target'],df['sex']).plot(kind='bar',figsize=(10,6),color=['salmon','lightblue'])


# plt.title("Male and female Comparision")
# plt.legend(['Female','Male'])
# plt.xticks(rotation=0)
# plt.show()

# plt.scatter(df.age[df['target'] == 1],df.thalach[df['target'] == 1],color='red')
# plt.scatter(df.age[df['target'] == 0],df.thalach[df['target'] == 0],color='blue')

# plt.show()

## check the distribution of the age


# plt.hist(df['age'],20)

# plt.show()


## heart disease frequency for chest pain type
print("CrossTab")
print(pd.crosstab(df.cp,df.target))

## make the crosstab more visual

# pd.crosstab(df.cp,df.target).plot(kind='bar',figsize=(10,6))

# plt.show()

## make a correlation matrix
corr = df.corr()

print(corr)

# sns.heatmap(corr,annot=True,cmap='YlGnBu')

# plt.show()

#Split the data into features and labels

X = df.drop(columns='target',axis=1)

Y = df['target']

print(X)

print(Y)


x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=42,test_size=0.2)

# print(x_train)

# print(y_train)

# print(x_test)

# print(y_test)


## now we got our model divided into train and test sets,now we deploy the appropriate model and find the metrics

## we are going to try three different machine learning model
## 1. Logisitic Regression  2.KNN classiffier  3.random Forest 

## put models in dictionarys

log = LogisticRegression()
knn = KNeighborsClassifier()
random_forest = RandomForestClassifier()
decision = DecisionTreeClassifier()

models = {'Logistic Regression':log,'KNN':knn,'Random Forest':random_forest}


## create a function to fit and score models

def fit_score(models,x_train,x_test,y_train,y_test):
    '''
    Fits and evaluates the given machine learning models
    models:A dict of the different machine learning models
    '''

    ## set random seed
    np.random.seed(42)
    
    ## make a list to keep model scores
    model_scores = {}
    
    ## loop through models
    for name,model in models.items():
        model.fit(x_train,y_train)
        
        ## evaluate the model and append its scores to model_scores
        model_scores[name] = model.score(x_test,y_test)
        
    return model_scores

        

model_scores = fit_score(models,x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test)

print(model_scores)

## model comparision

model_compare = pd.DataFrame(model_scores,index=['Accuracy'])


print(model_compare)

## transpose the dataframe

print(model_compare.T)

# model_compare.T.plot.bar()

# plt.show()


##now we got a baseline model ... and we know a model's first predicitons aren't always what we should be based our next steps off

## lets look at what should we do to improve our model
# # Hyperparameter tuning
# # Feauture importance
# # Confusion matrix
# # Cross validation
# # Precision
# # Recall
# # F1 Score 
# # Classification Report 
# # Roc curve 
# # Area under the curve(AUC)

## Hyperparameter Tuning(By hand)
## google how to tune particular model

## lets tune KNN
train_scores = []
test_scores = []

## create a list of differnt values for knn_classification


neighbors = range(1,21)

print(neighbors)

knn1 = KNeighborsClassifier()

## loop through Different N Neighbors

for i in neighbors:
    knn1.set_params(n_neighbors=i)
    
    ## fit the algorithm
    knn1.fit(x_train,y_train)
    
    ## update the training scores list
    train_scores.append(knn1.score(x_train,y_train))
    
    ## update the test scores list
    test_scores.append(knn1.score(x_test,y_test))
    
print(train_scores)
print(test_scores)


plt.figure(1,figsize=(10,8))

plt.plot(neighbors,train_scores,label='Train Data')
plt.plot(neighbors,test_scores,label='Test Data')
plt.xticks(list(range(1,21)))

plt.xlabel(" No of Neighbors")
plt.ylabel("Model score")

plt.legend(loc=0)

plt.show()


## hyperparameter tuning with randomized search cv

## we are going to tune 1. Logistic Regression() 2.RandomForestClassifier() .. using Randomized Search CV

## create a hyperparameter grid for logistic Regression

log_reg_grid = {"C":np.logspace(-4,4,50),'solver':['liblinear']}

## create a hyperparameter grid for randomforest classifer

random_for_grid = {"n_estimators":np.arange(10,1000,50),"max_depth":[None,3,5,10],'min_samples_split':np.arange(2,20,2),'min_samples_leaf':np.arange(1,20,2)}


## now we've got hyperparameters grid setup for each of our models,lets tune them using RandomizedSearchCV

## tune logistic regression

np.random.seed(42)

## setup random hyperparameter search for LogisticRegressio

rs_log_reg =  RandomizedSearchCV(LogisticRegression(),param_distributions=log_reg_grid,cv=5,n_iter=20,verbose=True)

## fit the random hyperparameter search for LogisticRegression

rs_log_reg.fit(x_train,y_train)


## checking the best parameters

print(rs_log_reg.best_params_)

## evaluate the model
print(rs_log_reg.score(x_test,y_test))

## now we've tuned Logisitc Regression,lets do the same for random forest classifier

## setup Random Seed

np.random.seed(42)

## setup random hyperparameter search for randomforest classifier

# rs_rf = RandomizedSearchCV(RandomForestClassifier(),param_distributions = random_for_grid,cv=5,n_iter=20,verbose=True)

# rs_rf.fit(x_train,y_train)

# ## finding the best parameters
# print(rs_rf.best_params_)

# # evaulate the model

# print(rs_rf.score(x_test,y_test))


## steps for hyperparameter tuning

##1.BY hand
## 2. RandomSearch CV
## 3.Grid Search CV

## hyperparamer tuning using Gridsearch CV

## since our Logistic Regression Model Provides the best scores so far,we'll try and improve them again using gridsearchCV

log_reg_grid1 = {"C":np.logspace(-4,4,30),'solver':['liblinear']}

## setup grid parameter search for logistic Regression

gs_log_reg = GridSearchCV(LogisticRegression(),log_reg_grid1,cv=5,verbose=True)


## fit
gs_log_reg.fit(x_train,y_train)

## check the best hyperparameters

print(gs_log_reg.best_params_)

##evaluate

print(gs_log_reg.score(x_test,y_test))


## evaluating our tuned machine learning classifier,Beyond Accuracy

## ROC Curve and AUC Score
## confusion Matrix
## classification report
## precision,recall,f1 score
## and it would be great if cross-validation was used wherever possible

## to make comparisions and evaluate our trained model,first we need to make predictions.

## make predictions with tuned model
y_preds = gs_log_reg.predict(x_test)

print(len(y_test == y_preds))

print((y_test == y_preds).sum())


## import Roc curve fun from sklearn.metrics and Calculate AUC

roc_curve = RocCurveDisplay.from_estimator(gs_log_reg,x_test,y_test,)


plt.show()

## confusion Matrix

conf = confusion_matrix(y_test,y_preds)

ConfusionMatrixDisplay(conf).plot()


plt.show()

## lets get classification report as well as cross validated precision,recall and f1 score

print(classification_report(y_test,y_preds))

## calculate evaluation metrics using cross validation

## were going to calculate precision,recall and f1 score using cross validation and to do so
## well be using cross val score

## check best hyper parameters

print(gs_log_reg.best_params_)

clf = LogisticRegression(C=0.20433597178569418,solver ='liblinear')


## cross validated accuracy
cross_val_acc = cross_val_score(clf,X,Y,cv=5,scoring='accuracy')

print(np.mean(cross_val_acc))

# cross validated precision

cross_val_precision = cross_val_score(clf,X,Y,cv=5,scoring='precision')

print(np.mean(cross_val_precision))

# cross validated recall

cross_val_recall = cross_val_score(clf,X,Y,cv=5,scoring='recall')

print(f'{np.mean(cross_val_recall)}')

# cross validated f1_score

cross_val_f1score = cross_val_score(clf,X,Y,cv=5,scoring='f1')


print(np.mean(cross_val_f1score))

score = []

metrics = ['accuracy','precision','recall','f1']


def scores(model,x,y):
    for i in metrics:
        score.append(np.mean(cross_val_score(model,x,y,cv=5,scoring=i)))
        
        
    return score
        
print(scores(clf,X,Y))

plt.bar(['accuracy','precision','recall','f1score'],[0.8479781420765027, 0.8215873015873015, 0.9272727272727274, 0.8705403543192143],align='edge')

plt.show()

## feauture importance is another way of asking which feautures contributed to most of the outcomes of the model and how did they contribute

## lets find our feature importance for logistic regression model

## fit an instance of logistic regression
    
clf.fit(x_train,y_train)

## check coef and intercept

print(clf.coef_)

print(clf.intercept_)


## create a dictionary of the coefficients

coeff = dict(zip(df.columns,clf.coef_[0]))

print(coeff)

plt.bar(df.columns.values[:-1],clf.coef_[0])

plt.show()

## save the model

dump(clf,'LinearRegressionHeart(Best Params)',)

# ##loading the model
# model_name = load('Filename')

## experimentation

## could you collect more data
## could you try a better model ? Like Catboost or XgBoost
## Could you improve the current models?
## if your model is good enough (you have hit your evaluation metrics) you can go ahead and save the model

