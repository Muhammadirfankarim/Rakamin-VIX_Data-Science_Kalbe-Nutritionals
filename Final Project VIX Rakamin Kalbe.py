#!/usr/bin/env python
# coding: utf-8

# ### Name : Muhammad Irfan Karim
# #### SRIWIJAYA UNIVERSITY

# ### Import Important Library

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# ### Making path for dataset

# In[2]:


nama_dataset1 = 'Customer'
nama_dataset2 = 'Product'
nama_dataset3 = 'Store'
nama_dataset4 = 'Transaction'
path1 = f'Case Study - {nama_dataset1}.csv'
path2 = f'Case Study - {nama_dataset2}.csv'
path3 = f'Case Study - {nama_dataset3}.csv'
path4 = f'Case Study - {nama_dataset4}.csv'


# ### Reading all 4 data

# In[3]:


df_customer = pd.read_csv(f'{path1}')


# In[4]:


df_customer.head()


# In[5]:


df_product= pd.read_csv(f'{path2}')


# In[6]:


df_product


# In[7]:


df_store = pd.read_csv(f'{path3}')


# In[8]:


df_store


# In[9]:


df_transaction = pd.read_csv(f'{path4}')


# In[10]:


df_transaction


# ## Data Cleaning

# #### 1.Customer Data

# In[11]:


df_customer.describe()


# In[12]:


df_customer.info()


# In[13]:


df_customer.isna().sum()


# In[14]:


df_customer['Income'] = df_customer['Income'].str.replace(',', '.')


# In[15]:


# dapat dilihat diatas bahwa tipe data dari income adalah object seharusnya tipe data nya berupa float
# mari kita lakukan transformasi data pada kolom income untuk mengubah tipe data
df_customer['Income'] = df_customer['Income'].astype(float)
# Setelah diubah lakukan pengecekan kembali terhadap tipe data pada kolom Income
df_customer.info()


# #### 2.Product Data

# In[16]:


df_product.describe()


# In[17]:


df_product.info()


# In[18]:


df_product.isna().sum()


# #### 3.Store Data

# In[19]:


df_store.describe()


# In[20]:


df_store.info()


# In[21]:


df_store['Latitude'] = df_store['Latitude'].str.replace(',', '')
df_store['Longitude'] = df_store['Longitude'].str.replace(',', '')


# In[22]:


#ubah tipe data object menjadi float
df_store['Latitude']  = df_store['Latitude'].astype(float)
df_store['Longitude'] = df_store['Longitude'].astype(float)
# Setelah diubah lakukan pengecekan kembali terhadap tipe data pada kolom Income
df_store.info()


# In[23]:


df_store.isna().sum()


# #### 4.Transaction Data

# In[24]:


df_transaction.describe()


# In[25]:


df_transaction.head()


# In[26]:


# Membuat visualisasi untuk melihat persebaran product id
sns.set(style='whitegrid')

# Membuat histogram menggunakan Seaborn
sns.histplot(data=df_transaction, x='ProductID', bins=10, kde=False, color='seagreen')
plt.xlabel('ProductID')
plt.ylabel('Frequency')
plt.title('Distribution of ProductID')
plt.show()


# In[27]:


df_transaction.info()


# In[28]:


# Mengubah tibe data date dari object menjadi datetime[64ns]
df_transaction['Date'] = df_transaction['Date'].astype('datetime64[ns]')
df_transaction.info()


# In[29]:


df_transaction.isna().sum()


# ### Data Merge

# In[30]:


# Merge df_transaction with df_customer using CustomerID
merged_df = df_transaction.merge(df_customer, on='CustomerID', how='left')

# Merge merged_df with df_product using ProductID
merged_df = merged_df.merge(df_product, on='ProductID', how='left')

# Merge merged_df with df_store using StoreID
merged_df = merged_df.merge(df_store, on='StoreID', how='left')

# Now you have a single dataframe that combines information from all four dataframes
display(merged_df)


# In[31]:


merged_df.head()


# In[32]:


merged_df.tail()


# In[33]:


merged_df.info()


# ### Making Regression time series Arima (Autoregressive Integrated Moving Average) with Scalecast 

# In[45]:


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from scalecast.Forecaster import Forecaster
from pmdarima import auto_arima


# In[48]:


# Group by 'Date' and aggregate 'Qty' using sum
regression_data = merged_df.groupby('Date')['Qty'].sum().reset_index()

# Print the resulting dataframe
regression_data


# In[47]:


# Set 'Date' column as the index
regression_data.set_index('Date', inplace=True)


# In[51]:


sns.set(rc={'figure.figsize':(14,7)})

f = Forecaster(y=regression_data['Qty'],current_dates=regression_data['Date'])

f.generate_future_dates(12) # 12-month forecast horizon
f.set_test_length(.2) # 20% test set
f.set_estimator('arima') # set arima
f.manual_forecast(call_me='arima1') # forecast with arima

f.plot_test_set(ci=True) # view test results
plt.title('ARIMA Test-Set Performance',size=14)
plt.show()


# ### Sepertinya hasil dari forecast terlihat kurang memuaskan maka dari itu kita akan menggunakan grid search untuk mencari hasil yang terbaik

# ## Grid search the optimal orders in scalecast

# In[54]:


f.set_validation_length(12)
grid = {
    'order':[(1,1,1),(1,1,0),(0,1,1)],
    'seasonal_order':[(2,1,1,12),(1,1,1,12),(2,1,0,12),(0,1,0,12)]
}

f.ingest_grid(grid)
f.tune()
f.auto_forecast(call_me='arima2')

f.plot_test_set(ci=True,models='arima2')
plt.title('ARIMA Test-Set Performance',size=14)
plt.show()

f.plot(ci=True,models='arima2')
plt.title('ARIMA Forecast Performance',size=14)
plt.show()

f.regr.summary()


# ## Clustering with KMeans

# In[41]:


# Group by 'Date' and aggregate 'Qty' using sum
regression_data = merged_df.groupby('Date')['Qty'].sum().reset_index()

# Print the resulting dataframe
regression_data


# In[42]:


merged_df.columns


# In[43]:


# Melakukan operasi grouping dan agregasi
clustering_data = merged_df.groupby('CustomerID').agg({
    'TransactionID': 'count',
    'Qty': 'sum',
    'TotalAmount': 'sum'
}).reset_index()

clustering_data


# In[44]:


clustering_data.info()


# In[45]:


plt.figure(figsize=(18,3))

plt.subplot(1,3,1)
sns.histplot(clustering_data['Qty'], color='royalblue', kde= True)
plt.title('Distribusi Quantity', fontsize=16)
plt.xlabel('Quantity', fontsize=14)
plt.ylabel('Sum', fontsize=14)

plt.subplot(1,3,2)
sns.histplot(clustering_data['TransactionID'], color='deeppink', kde= True)
plt.title('distribusi transaksi pelanggan', fontsize=16)
plt.xlabel('transaksi pelanggan', fontsize=14)


plt.subplot(1,3,3)
sns.histplot(clustering_data['TotalAmount'], color='seagreen', kde= True)
plt.title('distribusi Jumlah total', fontsize=16)
plt.xlabel('Total Amount', fontsize=14)
plt.ylabel('Sum', fontsize=14)

#plt.tight_layout()

plt.show()


# In[46]:


X1=clustering_data[['Qty', 'TransactionID', 'TotalAmount']]
wcss=[]
for n in range(1, 11):
    model1=KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=100)
    model1.fit(X1)
    wcss.append(model1.inertia_)
print(wcss)


# In[47]:


plt.figure(figsize=(8,3))
plt.plot(list(range(1, 11)), wcss, color='royalblue', marker='o', linewidth=2, markersize=12, markerfacecolor='m', markeredgecolor='m')
plt.title('WCSS vs Banyaknya Cluster', fontsize=18)
plt.xlabel('Jumlah Cluster', fontsize=15)
plt.ylabel('WCSS', fontsize=15)
plt.show()
#didapatkan menggunakan elbow method bahwa nilai k terbaik adalah 3


# In[48]:


# Mengambil fitur yang akan digunakan
features = clustering_data[['Qty', 'TransactionID', 'TotalAmount']]

# Normalisasi data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Jumlah kluster yang diinginkan
n_clusters = 3

# Melatih model KMeans
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=42)
clustering_labels = kmeans.fit_predict(scaled_features)
centroids = kmeans.cluster_centers_

# Menambahkan kolom Cluster ke DataFrame
clustering_data['Cluster'] = clustering_labels

# Menampilkan hasil klasterisasi
display(clustering_data)


# In[49]:


plt.figure(figsize=(10, 6))
plt.scatter(clustering_data['TransactionID'], clustering_data['CustomerID'], c=clustering_data['Cluster'], cmap='viridis')
plt.xlabel('TransactionID')
plt.ylabel('CustomerID')
plt.title('Scatter Plot of Clusters TransactionID vs CustomerID')
plt.colorbar(label='Cluster')
plt.show()


# In[50]:


plt.figure(figsize=(10, 6))
plt.scatter(clustering_data['Qty'], clustering_data['CustomerID'], c=clustering_data['Cluster'], cmap='viridis')
plt.xlabel('Qty')
plt.ylabel('CustomerID')
plt.title('Scatter Plot of Clusters Qty vs CustomerID')
plt.colorbar(label='Cluster')
plt.show()


# In[51]:


plt.figure(figsize=(10, 6))
plt.scatter(clustering_data['TotalAmount'], clustering_data['CustomerID'], c=clustering_data['Cluster'], cmap='viridis')
plt.xlabel('TotalAmount')
plt.ylabel('CustomerID')
plt.title('Scatter Plot of Clusters TotalAmount vs CustomerID')
plt.colorbar(label='Cluster')
plt.show()


# In[52]:


plt.figure(figsize=(10, 6))
plt.scatter(clustering_data['Qty'], clustering_data['TotalAmount'], c=clustering_data['Cluster'], cmap='viridis')
plt.xlabel('Qty')
plt.ylabel('TotalAmount')
plt.title('Scatter Plot of Clusters Qty vs Total Amount')
plt.colorbar(label='Cluster')
plt.show()


# ## Thankyou for reading!!
# ### By Muhammad Irfan Karim
