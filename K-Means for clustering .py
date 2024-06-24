#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the data
data = pd.read_csv('customers.csv')

# Check for missing values
print(data.isnull().sum())

# Fill missing values or drop them
data.dropna(inplace=True)

# Encode categorical variables
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_cols)

# Scale numerical variables
numerical_cols = data.select_dtypes(include=['number']).columns
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Principal Component Analysis (PCA)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# Plot the PCA results
plt.figure(figsize=(10, 7))
plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Customer Data')
plt.show()

# Determine optimal number of clusters using elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_pca)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 7))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method For Optimal k')
plt.show()

# Determine optimal number of clusters using silhouette score
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_pca)
    score = silhouette_score(data_pca, kmeans.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(10, 7))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Optimal k')
plt.show()

# Apply K-means with the chosen number of clusters
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(data_pca)

# Add PCA results for plotting
data['PC1'] = data_pca[:, 0]
data['PC2'] = data_pca[:, 1]

# Plot the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=data, palette='viridis')
plt.title('K-means Clustering with PCA')
plt.show()

# Analyze the clusters
columns_to_exclude = ['CustomerID', 'PC1', 'PC2']
numerical_cols = [col for col in data.columns if col not in columns_to_exclude and col != 'Cluster']

# Re-analyze the clusters
cluster_analysis = data.groupby('Cluster')[numerical_cols].mean()
print(cluster_analysis)


# In[ ]:




