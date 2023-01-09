#!/usr/bin/env python
# coding: utf-8

# In[458]:


#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[636]:


#Loading the dataset
df = pd.read_csv('application_data.csv')
df.head(5)


# In[393]:


#Sumary of the dataset
df.shape


# In[394]:


df.info()


# In[396]:


df.describe()


# In[465]:


#Cleaning the dataset
##Listing the columns having null values more than 30%
nullcol=df.isnull().sum()
nullcol=nullcol[nullcol.values>(0.3*len(nullcol))]
len(nullcol)


# In[466]:


##Removing the columns
nullcol = list(nullcol[nullcol.values>=0.3].index)
df.drop(labels=nullcol,axis=1,inplace=True)
print(len(nullcol))


# In[637]:


##Checking the updated dataset for null values
df.isnull().sum()/len(df)*100


# In[638]:


##Update the dataset missing values with median
median_value=df['AMT_ANNUITY'].median()
df.loc[df['AMT_ANNUITY'].isnull(),'AMT_ANNUITY']=median_value
df.isnull().sum()


# In[639]:


##Listing the rows having null values more than 30%
nullrow=df.isnull().sum(axis=1)
print(len(nullrow))


# In[640]:


##Removing the rows
nullrow=list(nullrow[nullrow.values>=0.3*len(df)].index)
df.drop(labels=nullrow,axis=0,inplace=True)
print(len(nullrow))


# In[469]:


##Accessing the 'XNA' values
df[df['CODE_GENDER']=='XNA'].shape


# In[163]:


##Accessing the values of Gender Column
df['CODE_GENDER'].value_counts()


# In[164]:


##Updating the column with "F"values
df.loc[df['CODE_GENDER']=='XNA','CODE_GENDER']='F'
df['CODE_GENDER'].value_counts()


# In[649]:


## Division the dataset as per the target group 
##Target=0(Clients with Non-payment Difficulties) and Target=1(client with payment difficulties)
target0_df=df.loc[df["TARGET"]==0]
target1_df=df.loc[df["TARGET"]==1]


# In[650]:


#Binning of Variables such as Amt_Income_Range and Amount_Credit
income_bins=[0,100000,200000, 500000,750000,10000000000]
df['AMT_INCOME_RANGE'] = pd.cut(df['AMT_INCOME_TOTAL'], bins = income_bins, labels=['VERY_LOW', 'LOW', "MEDIUM", 'HIGH', 'VERY_HIGH'])
df['AMT_INCOME_RANGE'].head(20)


# In[651]:


credit_bins = [0,100000,200000, 500000,750000,10000000000]
df['AMT_CREDIT_RANGE'] = pd.cut(df['AMT_CREDIT'], bins = income_bins, labels=['VERY_LOW', 'LOW', "MEDIUM", 'HIGH', 'VERY_HIGH'])
df['AMT_CREDIT_RANGE'].head(20)


# In[478]:


#Analysis
##Detecting Outliers
sns.boxplot(y =df['AMT_INCOME_TOTAL']).set(title="Distribution of AMT_INCOME_TOTAL")
plt.show()


# In[479]:


sns.boxplot(y =df['AMT_CREDIT']).set(title="Distribution of AMT_INCOME_TOTAL")
plt.show()


# In[657]:


#Checking the Data Imbalance 
df["TARGET"].value_counts().plot.pie(title='Target Imbalance Distribution',autopct='%1.0f%%')
plt.show()


# In[170]:


##Calculating Imbalance percentage
round(len(target0_df)/len(target1_df),2)


# In[612]:


#Univariate Analysis of Categorical Variables
##Distribution of Income Range 
df['AMT_CREDIT_RANGE'].value_counts().plot.bar(title='Distribution of Income Range')
plt.show()


# In[494]:


# Gender Distibution of Loan Non-Payment Difficulties
target0_df["CODE_GENDER"].value_counts().plot.pie(title='Gender Distibution of Loan- Non Payment Difficulties',autopct='%1.0f%%')
plt.show()


# In[504]:


# Gender Distibution of Loan Payment Difficulties
target1_df["CODE_GENDER"].value_counts().plot.pie(title='Gender Distibution of Loan Payment Difficulties',autopct='%1.0f%%')
plt.show()


# In[595]:


# Income sources of Loan- Non Payment Difficulties
plt.figure(figsize=(7,7))
target0_df["NAME_INCOME_TYPE"].value_counts().plot.pie(title ='Income sources of Loan- Non Payment Difficulties',autopct='%1.0f%%')
plt.show()


# In[596]:


#Income sources of Loan Payment Difficulties
plt.figure(figsize=(7,7))
target1_df["NAME_INCOME_TYPE"].value_counts().plot.pie(title ='Income sources of Loan Payment Difficulties',autopct='%1.0f%%')
plt.show()


# In[597]:


#Family Status of Loan- Non Payment Difficulties
plt.figure(figsize=(5,5))
target0_df["NAME_FAMILY_STATUS"].value_counts().plot.pie(title= 'Family Status of Loan- Non Payment Difficulties',autopct='%1.0f%%')
plt.show()


# In[598]:


#Family Status of Loan Payment Difficulties
plt.figure(figsize=(5,5))
target1_df["NAME_FAMILY_STATUS"].value_counts().plot.pie(title= 'Family Status of Loan Payment Difficulties',autopct='%1.0f%%')
plt.show()


# In[600]:


#Education Status of Loan-Non Payment Difficulties
plt.figure(figsize=(7,7))
target0_df["NAME_EDUCATION_TYPE"].value_counts().plot.pie(title= 'Education Status of Loan-Non Payment Difficulties',autopct='%1.0f%%')
plt.show()


# In[601]:


#Education Status of Loan Payment Difficulties
plt.figure(figsize=(5,5))
target1_df["NAME_EDUCATION_TYPE"].value_counts().plot.pie(title= 'Education Status of Loan Payment Difficulties',autopct='%1.0f%%')
plt.show()


# In[602]:


#Type of House of Loan-Non Payment Difficulties
plt.figure(figsize=(5,5))
target0_df['NAME_HOUSING_TYPE'].value_counts().plot.pie(title= 'Type of House of Loan-Non Payment Difficulties',autopct='%1.0f%%')
plt.show()


# In[603]:


#Type of House of Loan Payment Difficulties
plt.figure(figsize=(5,5))
target1_df['NAME_HOUSING_TYPE'].value_counts().plot.pie(title= 'Type of House of Loan Payment Difficulties',autopct='%1.0f%%')
plt.show()


# In[654]:


#Income range of Loan-Non Payment Difficulties
plt.figure(figsize=(5,5))
target0_df['AMT_INCOME_RANGE'].value_counts().plot.pie(title= 'Income range of Loan-Non Payment Difficulties',autopct='%1.0f%%')
plt.show()


# In[655]:


#Income range of Loan Payment Difficulties
plt.figure(figsize=(5,5))
target1_df['AMT_INCOME_RANGE'].value_counts().plot.pie(title= 'Income range of Loan Payment Difficulties',autopct='%1.0f%%')
plt.show()


# In[630]:


#Types of Loans taken by Loan-Non Payment Difficulties
target0_df['NAME_CONTRACT_TYPE'].value_counts().plot.pie(title= 'Types of Loans taken by Loan-Non Payment Difficulties',autopct='%1.0f%%')
plt.show()


# In[226]:


#Types of Loans taken by Loan Payment Difficulties
target1_df['NAME_CONTRACT_TYPE'].value_counts().plot.pie(title= 'Types of Loans taken by Loan Payment Difficulties',autopct='%1.0f%%')
plt.show()


# In[262]:


#Types of Organizations who applied for loan - Non-Payment Difficulties
target0_df["ORGANIZATION_TYPE"].value_counts().plot.bar(title= 'Types of Organizations who applied for loan - Non-Payment Difficulties', figsize=(15,4))
plt.show()


# In[260]:


#Types of Organizations who applied for loan Payment Difficulties
target1_df["ORGANIZATION_TYPE"].value_counts().plot.bar(title= 'Types of Organizations who applied for loan Payment Difficulties', figsize=(15,4))
plt.show()


# In[511]:


##Finidng Outliers as per the Dataset Divison
##Target=0(Clients with Non-payment Difficulties) and Target=1(client with payment difficulties)
sns.boxplot(y = target0_df['AMT_ANNUITY']).set(title="Distribution of Annuity Amount For Target-0")
plt.show()


# In[512]:


sns.boxplot(y = target1_df['AMT_ANNUITY']).set(title="Distribution of Annuity Amount For Target-1")
plt.show()


# In[299]:


sns.boxplot(y = target0_df['AMT_INCOME_TOTAL']).set(title="Distribution of Income Amount For Target-0")
plt.show()


# In[302]:


sns.boxplot(y = target1_df['AMT_INCOME_TOTAL']).set(title="Distribution of Income Amount For Target-1")
plt.show()


# In[513]:


sns.boxplot(y = target0_df['AMT_CREDIT']).set(title="Distribution of Credit Amount For Target-0")
plt.show()


# In[304]:


sns.boxplot(y = target1_df['AMT_CREDIT']).set(title="Distribution of Credit Amount For Target-1")
plt.show()


# In[533]:


##Bivariate analysis for numerical variables
##Box plotting for Income amount vs Education Status For target-0
plt.figure(figsize=(15,10))
plt.yscale('log')
sns.boxplot(data =target0_df, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v').set(title='Income amount vs Education Status for Target-0')
plt.show()


# In[535]:


##Box plotting for Income amount vs Education Status for Target-1
plt.figure(figsize=(15,10))
plt.yscale('log')
sns.boxplot(data =target1_df, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v').set(title='Income amount vs Education Status for Target-1')
plt.show()


# In[534]:


##Box plotting for Credit amount vs Education Status for Target-0
plt.figure(figsize=(15,10))
sns.boxplot(data =target0_df, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient='v').set(title='Credit amount vs Education Status for Target-0')
plt.show()


# In[536]:


##Box plotting for Credit amount vs Education Status for Target-1
plt.figure(figsize=(15,10))
sns.boxplot(data =target1_df, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient='v').set(title='Credit Amount vs Education Status for Target-1')
plt.show()


# In[631]:


##Bivariate Analysis of Numerical vs Numerical Variables for Target-0
graph= target0_df[['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL','DAYS_BIRTH']].fillna(0)
sns.pairplot(graph).set(title='Pairplot for Target-0')
plt.show()


# In[632]:


##Bivariate Analysis of Numerical vs Numerical Variables for Target-1
graph = target1_df[['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL','DAYS_BIRTH']].fillna(0)
sns.pairplot(graph).set(title='Pairplot for Target-1')
plt.show()


# In[539]:


##Heatmap of Target-0
plt.figure(figsize=(12, 10))
sns.heatmap(target0_df[['AMT_INCOME_TOTAL','AMT_ANNUITY','DAYS_EMPLOYED','DAYS_BIRTH','DAYS_REGISTRATION', 'DAYS_ID_PUBLISH','AMT_CREDIT',]].corr(method = 'pearson'),cmap="RdYlGn",annot=False).set(title='Heatmap of Target-0')
plt.show()


# In[542]:


##Heatmap of Target-1
plt.figure(figsize=(12, 10))
sns.heatmap(target1_df[['AMT_INCOME_TOTAL','AMT_ANNUITY','DAYS_EMPLOYED','DAYS_BIRTH','DAYS_REGISTRATION', 'DAYS_ID_PUBLISH','AMT_CREDIT',]].corr(method = 'pearson'),cmap="RdYlGn",annot=False).set(title='Heatmap of Target-1')
plt.show()


# In[371]:


#DataAnalysis of Previous Application Dataset
##Loading the dataset
df_prev= pd.read_csv('previous_application.csv')
df_prev.head(10)


# In[375]:


##Sumary of the dataset
df_prev.shape


# In[376]:


df_prev.info()


# In[408]:


df_prev.describe()


# In[543]:


#Cleaning the dataset
##Accessing the 'XNA' values
df_prev=df_prev.replace('XNA', np.NaN)
df_prev=df_prev.replace('XAP', np.NaN)


# In[407]:


df_prev['NAME_CONTRACT_STATUS'].value_counts()


# In[550]:


##Distribution of Contract Status of Previous Application
plt.figure(figsize=(7,7))
df_prev["NAME_CONTRACT_STATUS"].value_counts().plot.pie(title='Contract status of previous application',autopct='%1.0f%%')
plt.show()


# In[554]:


##Distribution of purposes for Loan
plt.figure(figsize=(7,7))
df_prev["NAME_CASH_LOAN_PURPOSE"].value_counts().plot.barh(title='Distribution of purposes for Loan')
plt.show()


# In[418]:


##Day on which loan was applied
df_prev["WEEKDAY_APPR_PROCESS_START"].value_counts().plot.barh(title='Day on which loan was applied')
plt.show()


# In[417]:


##Reason for Loan Rejection
df_prev["CODE_REJECT_REASON"].value_counts().plot.barh(title='Reason for Loan Rejection')
plt.show()


# In[658]:


##Distribution of Type of Client
df_prev["NAME_CLIENT_TYPE"].value_counts().plot.pie(title='Distribution of Type of Client',autopct='%1.0f%%')
plt.show()


# In[445]:


### Merging the Application dataset with Previous appliaction dataset
com_df=pd.merge(left=df,right=df_prev,how='inner',on='SK_ID_CURR')
com_df.shape


# In[450]:


##Merged Data
com_df = df.merge(df_prev,on='SK_ID_CURR', how='inner')
com_df.head(5)


# In[586]:


##Prev Credit amount vs Housing type
plt.figure(figsize=(16,12))
plt.xticks(rotation=90)
sns.barplot(data =com_df, y='AMT_CREDIT_x',hue='TARGET',x='NAME_CASH_LOAN_PURPOSE').set(title='Prev Credit amount vs Loan Purpose')
plt.show()


# In[582]:


##Prev Credit amount vs Housing type
plt.figure(figsize=(16,12))
sns.barplot(data =com_df, y='AMT_CREDIT_x',hue='TARGET',x='NAME_HOUSING_TYPE').set(title='Prev Credit amount vs Housing type')
plt.show()


# In[659]:


##Prev Credit amount vs Income type
plt.figure(figsize=(16,12))
sns.barplot(data =com_df, y='AMT_CREDIT_x',hue='TARGET',x='NAME_INCOME_TYPE').set(title='Prev Credit amount vs Income Type')
plt.show()


# In[ ]:




