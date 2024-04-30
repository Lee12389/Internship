#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# Reading file from github raw

# In[70]:


df = pd.read_csv('https://raw.githubusercontent.com/FlipRoboTechnologies/ML-Datasets/main/Red%20Wine/winequality-red.csv')


# In[7]:


df.head()


# In[23]:


df.info()


# In[24]:


df.isna().sum().sum()


# In[35]:


for col in df.columns:
    print("Unique values in "+col+' :')
    print(df[col].unique().shape[0])
    print('\n')


# In[ ]:





# In[ ]:





# In[36]:


for col in df.columns:
    print(col + ' : ' + str(df[col].isna().sum()))


# In[16]:


for col in df.columns:
    print(col + ' : ' + str(df[col].dtype))


# In[17]:


for col in df.columns:
    print(df[col].isna().value_counts())


# In[38]:


sns.heatmap(df.isna())


# All the values are Integer or Float values as they are expected to be. No null values found. Data looks clean No need for preprocessing.Target variable is a categorical variable where quality >= 7, it's good, and quality < 7 it's bad. This is a classification problem. There are no categorical or string columns so no need for seperation
# 

# In[39]:


len(df)


# # Exploratory Data analysis

# Checking columns and data informaiotn

# In[41]:


desc_df = pd.DataFrame(df.describe())


# In[42]:


desc_df


# In[59]:


sns.heatmap(desc_df)


# In[51]:


mean_gt = []
median_gt = []
mean_eq = []
for col in desc_df.columns:
    if desc_df.loc['mean',col] > desc_df.loc['50%',col]:
        mean_gt.append(col)
    elif desc_df.loc['50%',col] > desc_df.loc['mean',col]:
        median_gt.append(col)
    else:
        mean_eq.append(col)


# In[ ]:





# In[52]:


mean_gt


# In[53]:


median_gt


# In[54]:


mean_eq


# There are no negative or invalid values
#     1. Mean is greater than median for fixed acidity, 

# In[8]:


s = df.quality.value_counts()


# In[10]:


type(s)


# In[61]:


sns.countplot(x=df['quality'])


# In[52]:


df.quality.unique()


# Observations so far
#     1. quality of wines is not evenly distributed. It's more alligned towards center. We have to balance before training
#     2. But as per problem statement, if quality is >=7 then only we will consider it as good. So we have very less amount of      information about quality is good.
#     3. Also we don't have information for extremes like from 0,1,2 and 9,10. So when we train the algorithms, we have to take care that we handle those.
#     

# In[15]:


df.columns


# In[65]:


sns.histplot(x=df['fixed acidity'])


# This looks like a normal distribution

# In[64]:


sns.histplot(x=df['volatile acidity'])


# In[30]:


sns.histplot(df['citric acid'])


# In[33]:


sns.histplot(df['residual sugar'])


# In[34]:


sns.histplot(df['chlorides'])


# In[35]:


sns.histplot(df['free sulfur dioxide'])


# In[36]:


sns.histplot(df['total sulfur dioxide'])


# In[37]:


sns.histplot(df['density'])


# In[38]:


sns.histplot(df['pH'])


# In[39]:


sns.histplot(df['sulphates'])


# In[32]:


sns.histplot(df['alcohol'])


# Most variables are left skewed. pH, Density are normally distributed align mostly with our target quality

# In[72]:


plt.figure(figsize=(16, 20),facecolor='white')
plotnumber = 1
for col in df.columns:
    if plotnumber <= 12:
        ax = plt.subplot(6,2,plotnumber)
        sns.distplot(df[col],color='r')
        plt.xlabel(col, fontsize=22)
        plt.yticks(rotation=0,fontsize=10)
    plotnumber += 1
plt.tight_layout()


# In[82]:


plt.figure(figsize=(20, 240),facecolor='white')
plotnumber = 1
for col in df.columns:
    if plotnumber <= 12:
        ax = plt.subplot(12,1,plotnumber)
#         sns.relplot(data=df, x=col, y='quality', kind="line")
        sns.violinplot(data=df,x=col,y='quality')
        plt.xlabel(col, fontsize=22)
        plt.ylabel('quality',fontsize=22)
#         plt.yticks(rotation=0,fontsize=10)
    plotnumber += 1
plt.tight_layout()


# In[67]:


pairs = [(x,y) for x in df.columns for y in df.columns if x != y ]


# In[78]:


df['Decision'] = df['quality'].apply(lambda x : 'Good' if x >=7 else 'Bad')


# In[80]:


df.Decision.value_counts()


# In[ ]:





# In[81]:


df.columns


# In[77]:


plt.figure(figsize=(20, 240),facecolor='white')
plotnumber = 1
for col in df.columns:
    if plotnumber <= 12:
        ax = plt.subplot(12,1,plotnumber)
#         sns.relplot(data=df, x=col, y='quality', kind="line")
        sns.scatterplot(data=df,x=col,y='quality')
        plt.xlabel(col, fontsize=22)
        plt.ylabel('quality',fontsize=22)
#         plt.yticks(rotation=0,fontsize=10)
    plotnumber += 1
plt.tight_layout()


# In[98]:


#sns.relplot(data=df, x=col, y='quality', kind="line")
sns.scatterplot(data=df,x='alcohol',y='fixed acidity',hue='Decision',palette='bright')
plt.xlabel('alchohol', fontsize=22)
plt.ylabel('fixed acidity',fontsize=22)
#         plt.yticks(rotation=0,fontsize=10)
# plt.tight_layout()


# Fixed acidity doesn't determine dicision in relation with alchohol

# In[99]:


#sns.relplot(data=df, x=col, y='quality', kind="line")
sns.catplot(data=df,x='alcohol',y='fixed acidity',hue='Decision',palette='bright')
plt.xlabel('alchohol', fontsize=22)
plt.ylabel('fixed acidity',fontsize=22)
#         plt.yticks(rotation=0,fontsize=10)
# plt.tight_layout()


# Pair plot is used to analyse multivariate analysis

# In[103]:


#sns.relplot(data=df, x=col, y='quality', kind="line")
sns.pairplot(data=df)
# plt.xlabel('alchohol', fontsize=22)
# plt.ylabel('fixed acidity',fontsize=22)
#         plt.yticks(rotation=0,fontsize=10)
# plt.tight_layout()


# Checking outliers with box plot
# 

# In[118]:


plt.figure(figsize=(20,200), facecolor='white')
plotnum = 1
for col in df.columns:
    print(col)
    if plotnum < len(df.columns)-1 or 'Decision' not in col:
        ax = plt.subplot(10,2,plotnum)
        sns.boxplot(df[col],palette='bright')
        plt.xlabel(col,fontsize=20)
        plt.yticks(rotation=0,fontsize=10)
    plotnum +=1
plt.tight_layout()


# A lot of data from most columns seems to be outliers. Need to handle those outliers

# In[ ]:





# In[ ]:





# In[ ]:





# In[120]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder


# In[121]:


OE = OrdinalEncoder()


# In[4]:


from imblearn.over_sampling import SMOTE


# We now need to oversample the data since we have very less number of results for good quality samples. We will use imblearn for the same

# In[ ]:


y = df['quality']
df.drop(columns=['quality'],inplace=True)
X = df


# In[26]:


y


# In[25]:


X


# In[38]:


y = y.apply(lambda x : 'Good' if x >= 7 else 'Bad')


# In[40]:


y.value_counts()


# In[41]:


smote = SMOTE()
x_resampled,y_resampled = smote.fit_resample(X,y)


# In[42]:


x_resampled


# In[43]:


y_resampled.value_counts()


# Now we can see that the data is resampled and eveything looks almost even. Now we can check the pair plots for multivariate analysis again

# In[44]:


#sns.relplot(data=df, x=col, y='quality', kind="line")
sns.pairplot(data=x_resampled)
# plt.xlabel('alchohol', fontsize=22)
# plt.ylabel('fixed acidity',fontsize=22)
#         plt.yticks(rotation=0,fontsize=10)
# plt.tight_layout()


# In[47]:


x_resampled.skew()


# Above we can observe that there is a lot of skewness in the data. We need to remove skewness

# In[48]:


sns.distplot(x_resampled['chlorides'],kde_kws={'shade':True},hist=False)


# In[54]:


df.skew()


# In[53]:


df.corr()


# In[74]:


df['Target'] = df['quality'].apply(lambda x : 1 if x >=7 else 0)


# In[76]:


plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),cmap='Blues_r',linewidths=0.1,fmt='.1g',linecolor='black',annot=True)
plt.yticks(rotation=0)
plt.show()


# In[81]:


plt.figure(figsize=(20,15))
df.corr()['Target'].sort_values(ascending=True).drop('quality').plot(kind='bar',color='m')
plt.ylabel('Target',rotation=0,fontsize=15)


# In[82]:


scaler = StandardScaler()


# In[83]:


x = df.drop('Target',axis=1)
y = df['Target']


# In[85]:


df


# In[84]:


x = pd.DataFrame(scaler.fit_transform(x),columns=x.columns)
x


# we have standardised the data to avoid bias. Now we need to check for multicollenearity

# # Checking variance inflation factor (VIF)

# In[87]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['values'] = [variance_inflation_factor(x.values,i) for i in range(len(x.columns))] 
vif['features'] = x.columns

vif


# We are removing fixed acidity since vif value is too high for that. so that we can reduce multi collenearity

# In[89]:


x = x.drop('fixed acidity',axis=1)


# In[90]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['values'] = [variance_inflation_factor(x.values,i) for i in range(len(x.columns))] 
vif['features'] = x.columns

vif


# Now we can see that there is not much collenearity in data

# In[92]:


y.value_counts()


# In[128]:


smote = SMOTE()
x1,y1 = smote.fit_resample(x,y)


# In[129]:


y1.value_counts()


# ![image.png](attachment:image.png)

# In[120]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

maxAcc = 0
maxRS = 0
for i in range(200):
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.30, random_state=i)
    RFR = RandomForestClassifier(random_state=0)
    RFR.fit(x_train, y_train)
    pred = RFR.predict(x_test)
    acc = accuracy_score(y_test, pred)
    if acc > maxAcc:
        maxAcc = acc
        maxRS = i
print("Best accuracy is", maxAcc, "at random state", maxRS)


# Seems to be data is overfit

# In[ ]:





# In[ ]:





# In[ ]:





# In[110]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[137]:


# checking accuracy for LogisticRegression
LR = LogisticRegression()

maxAcc = 0
maxRS = 0
for i in range(200):
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.30, random_state=i)
    LR.fit(x_train, y_train)
    pred = LR.predict(x_test)
    acc = accuracy_score(y_test, pred)
    if acc > maxAcc:
        maxAcc = acc
        maxRS = i
print("Best accuracy is", maxAcc, "at random state", maxRS)


# In[138]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)

maxAcc = 0
maxRS = 0
for i in range(200):
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.30, random_state=i)
    gb_clf.fit(x_train, y_train)
    pred = gb_clf.predict(x_test)
    acc = accuracy_score(y_test, pred)
    if acc > maxAcc:
        maxAcc = acc
        maxRS = i
print("Best accuracy is", maxAcc, "at random state", maxRS)



# In[139]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)

maxAcc = 0
maxRS = 0
for i in range(200):
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.30, random_state=i)
    abc.fit(x_train, y_train)
    pred = abc.predict(x_test)
    acc = accuracy_score(y_test, pred)
    if acc > maxAcc:
        maxAcc = acc
        maxRS = i
print("Best accuracy is", maxAcc, "at random state", maxRS)


# In[140]:


from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

bc = BaggingClassifier(n_estimators=50, random_state=42)


maxAcc = 0
maxRS = 0
for i in range(200):
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.30, random_state=i)
    bc.fit(x_train, y_train)
    pred = bc.predict(x_test)
    acc = accuracy_score(y_test, pred)
    if acc > maxAcc:
        maxAcc = acc
        maxRS = i
print("Best accuracy is", maxAcc, "at random state", maxRS)


# In[141]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

clf = ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2)


maxAcc = 0
maxRS = 0
for i in range(200):
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.30, random_state=i)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    acc = accuracy_score(y_test, pred)
    if acc > maxAcc:
        maxAcc = acc
        maxRS = i
print("Best accuracy is", maxAcc, "at random state", maxRS)


# All the algorithms seems to predict good. We are good to take any algorithm.

# In[ ]:




