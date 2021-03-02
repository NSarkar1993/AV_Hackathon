#!/usr/bin/env python
# coding: utf-8

# In[24]:


#Import all required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import roc_auc_score, plot_roc_curve
import catboost
from catboost import CatBoostClassifier
import shap


# In[25]:


#Path to the hackathon folder
path = "D:\\DataScience\\Hackathon"


# In[26]:


#Set the file names
trainfile = "train_Df64byy.csv"
testfile = "test_YCcRUnU.csv"


# In[27]:


#Read the files
train = pd.read_csv(path+"\\"+trainfile)
test = pd.read_csv(path+"\\"+testfile)


# In[28]:


train.head() #Top 5 rows


# In[29]:


#Check the data types of each column
train.dtypes


# In[30]:


#Check the missing values in train and test
train.isnull().sum()


# In[31]:


test.isnull().sum()


# In[32]:


#Confirm if Holding Policy Type and Holding Policy have same indices of missing rows
np.where(train['Holding_Policy_Type'].isnull())[0].all()==np.where(train['Holding_Policy_Duration'].isnull())[0].all()


# In[33]:


#Functions to preprocess the data. Missing imputation strategy is given in details in the submitted doc
def data_preprocess(df):
    df['Health Indicator'] = df['Health Indicator'].fillna(value="XX")
    df['Holding_Policy_Type'] = df['Holding_Policy_Type'].fillna('0.0')
    df['Holding_Policy_Duration'] = df['Holding_Policy_Duration'].fillna('0')
    
#     Convert Holding type to object dtype for categorical
    df['Holding_Policy_Type'] = df['Holding_Policy_Type'].astype('str')
    df['Region_Code'] = df['Region_Code'].astype('str')
    df['Reco_Policy_Cat'] = df['Reco_Policy_Cat'].astype('str')
    return df


# In[34]:


train = data_preprocess(train)
train.head()


# In[35]:


train.describe() #Sanity check for numerical features


# In[36]:


key_cols = ['ID']
dep_var = 'Response'


# In[37]:


#Create quartile bins for Lower Age on train. Use the same bins for binning in test data
train['Lower_Age_Quartile'],bins = pd.qcut(train['Lower_Age'],4,labels=['Q1','Q2','Q3','Q4'],retbins=True)
train['Lower_Age_Quartile'] = train['Lower_Age_Quartile'].astype(str)

def create_lower_age_bins(df,bins):
    #To be used for creating similar bins on testing data
    df['Lower_Age_Quartile']=pd.cut(df['Lower_Age'],bins,
                                    labels = ['Q1','Q2','Q3','Q4'],
                                    include_lowest=True).astype(str)
    return df


# In[38]:


#Data visualization. Comparison of age and policy holder with responses
df = train.copy()
df['Upper_Age_Bins'] = pd.qcut(train['Upper_Age'],7,retbins=False)

groups = df.groupby(['Upper_Age_Bins','Holding_Policy_Type'], sort=True)[dep_var].sum()
plot = groups.unstack().plot(figsize=(10,10))
plot.set_ylabel("Number of responses")
# plot.show()


# In[39]:


#Based on QC, we see that age 21-29 are most likely to take an insurance given that they do not hold a policy
#Therefore, creating a dummy feature to capture this
def create_dum_features(df,isTest = True):
    if isTest==True:
        df = create_lower_age_bins(df,bins) #Create lower quartile bins if it's a test data
    
    def age_check(row):
        age = row['Upper_Age']
        if age>=21 and age<=29 and row['Holding_Policy_Type']=='0.0': 
            return '1'
        else:
            return '0'
    
    df['high_prob_age']=df.apply(lambda row:age_check(row), axis=1).astype(str)
    return df


# In[40]:


#Create dummy in train dataset
train = create_dum_features(train,isTest = False)
train.head()


# In[41]:


#Preprocessing test data and creating dummy features
test = data_preprocess(test)
test = create_dum_features(test,isTest = True)
test.head()


# In[42]:


train.head()


# In[43]:


#Extract the list of all categorical features
cat_cols = train.select_dtypes(include = ["object","category"]).columns.tolist()
cat_cols


# In[65]:


#Split the given train set to training and validation dataset 90:10
train_df, val_df = train_test_split(train, test_size=0.1, random_state=123)


# In[66]:


#Drop key columns and dependant variable from train set
drop_cols = key_cols+[dep_var]
X_train = train_df.drop(drop_cols,axis=1)
Y_train = train_df[dep_var]


# In[67]:


X_test = val_df.drop(drop_cols,axis=1)
Y_test = val_df[dep_var]


# In[68]:


len(X_train)


# In[69]:


len(X_test)


# In[70]:


#Check the distribution of labels in train and validation dataset
print("Train dataset event rate = ", Counter(Y_train)[1]/(Counter(Y_train)[0]+Counter(Y_train)[1])*100,"%")
print("Validation dataset event rate = ",Counter(Y_test)[1]/(Counter(Y_test)[0]+Counter(Y_test)[1])*100,"%" )


# In[50]:


#Setting up CatBoost parameters


# In[74]:


#In the interest of time, grid search was not performed. The following hyperparameters were selected basis manual tuning

params = {'loss_function':'Logloss', 
         'learning_rate':0.025, 
          'iterations':1500, 
         'depth':8, 
         'random_seed':123,
          'early_stopping_rounds':100,
         'boosting_type':'Ordered',
          'use_best_model':True,
         'eval_metric' : 'AUC'}


# In[75]:


cat_features = cat_cols #Setting categorical features

#Model training begins
model = CatBoostClassifier(**params)

model.fit(X_train, Y_train, cat_features, 
          eval_set = [(X_train, Y_train),(X_test, Y_test)],
          verbose=True, plot=True)


# In[76]:


#Predict probabilities of the positve class
y_pred = model.predict_proba(X_test)[:,1]


# In[77]:


#Check the ROC AUC score on validation dataset
roc_auc_score(Y_test, y_pred)


# In[78]:


#Plot roc curve
plot_roc_curve(model, X_test, Y_test)


# In[79]:


#Check the feature importance of our model
model.get_feature_importance(data=catboost.Pool(X_test, label=Y_test, cat_features=cat_cols),
                             type='FeatureImportance',prettified=True,verbose=True)


# In[80]:


#Check the interaction between features in the model
model.get_feature_importance(data=catboost.Pool(X_test, label=Y_test, cat_features=cat_cols),
                             type='Interaction',prettified=True,verbose=True)


# In[58]:


# create a SHAP dependence plot to show the effect of a single feature across the whole dataset
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(catboost.Pool(X_train, Y_train, cat_features=cat_features))
shap.dependence_plot("Reco_Policy_Cat", shap_values, X_train)


# In[59]:


#Check importance of all numerical and categorical features
shap.summary_plot(shap_values, X_train)


# In[60]:


#Save the model as cbm format
model.save_model(path+"\\model_final.cbm",format = 'cbm')


# In[81]:


#Create submission on given test dataset

test_id = test['ID']
test_x = test.drop(['ID'],axis=1)
pred = model.predict_proba(test_x)


# In[82]:


submission = pd.DataFrame(test_id).join(pd.DataFrame(pred[:,1]))


# In[83]:


submission.columns = ['ID','Response']
submission


# In[84]:


submission.to_csv(path+"\\"+"submission_final.csv", header=True, index=False)

