#!/usr/bin/env python
# coding: utf-8

# In[1]:
import streamlit as st
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay,classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.tree import export_graphviz
from IPython.display import Image
warnings.filterwarnings('ignore')
df=pd.read_csv('Amazon Sale Report.csv')
df.head(10)


# In[2]:


df.columns = df.columns.str.strip()

for col in ['Amount', 'Gross Amt', 'Rate']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace('[^0-9.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')


# In[3]:


df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0).astype(int)


# In[4]:


df.dropna(subset=['Amount', 'Qty', 'Category'], inplace=True)


# In[5]:


df.drop(columns= ['index','Unnamed: 22', 'fulfilled-by', 'ship-country', 'currency', 'promotion-ids', 'ship-postal-code','ship-state', 'ship-city' ], inplace = True)


# In[6]:


df.isnull().sum()


# In[7]:


df.drop(['Courier Status'],axis=1, inplace = True)


# In[8]:


df.isnull().sum()


# In[9]:


numeric_cols = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_cols.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
st.pyplot(plt.gcf())




# In[10]:


df['B2B'].replace(to_replace=[True,False],value=['business','customer'], inplace=True)


# In[11]:


df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y')


# In[12]:


df.head()


# In[13]:


df['year_month'] = pd.to_datetime(df['Date']).dt.to_period('M')

monthly_sales = df.groupby('year_month')['Amount'].sum().reset_index()
print(monthly_sales)

plt.figure(figsize= (10,6))
plt.bar(monthly_sales['year_month'].dt.strftime('%Y-%m'), monthly_sales['Amount'])

plt.xlabel('Month')
plt.ylabel('sales amount')
plt.title(' Monthly sales')

plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt.gcf())




# In[14]:


order_status_dist = df['Status'].value_counts()
plt.figure(figsize=(8, 6))
order_status_dist.plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Order Status Distribution')
st.pyplot(plt.gcf())




# In[15]:


cancelled_orders = df[df['Status'] == 'Cancelled']
cancelled_reasons = cancelled_orders.groupby('Category')['Qty'].count().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
cancelled_reasons.plot.bar(color='salmon')
plt.title('Reasons for Cancelled Orders')
plt.xlabel('Category')
plt.ylabel('Number of Cancelled Orders')
st.pyplot(plt.gcf())




# In[16]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='B2B', y='Amount', data=df, palette='Set2')
plt.title('B2B vs. Customer Sales')
plt.xlabel('Customer Type')
plt.ylabel('Amount')
st.pyplot(plt.gcf())




# In[17]:


size_sales = df.groupby('Size')['Amount'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
size_sales.plot.bar(color='skyblue')
plt.title('Size-wise Sales Analysis')
plt.xlabel('Size')
plt.ylabel('Total Sales')
st.pyplot(plt.gcf())




# In[18]:


x=df.drop(columns=["Amount","SKU","Order ID","Status","Fulfilment","Sales Channel","ship-service-level","Style","Category","Size","ASIN"])
y=df[["Amount"]]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[19]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
numeric_cols=x.select_dtypes(include=np.number).columns
x[numeric_cols] = scaler.fit_transform(x[numeric_cols])
x.head()


# In[20]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
# for col in x.select_dtypes(include=['object']).columns:
#     x[col] = label_encoder.fit_transform(x[col])
df['B2B'] = label_encoder.fit_transform(df['B2B'])
df.head()


# In[21]:


# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

# model.fit(X_train, y_train)

