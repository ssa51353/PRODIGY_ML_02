# Customer Segmentation Using K-Means and PCA

This project performs customer segmentation using K-Means clustering and Principal Component Analysis (PCA). By analyzing customer data, we can identify distinct groups of customers and gain insights into their behaviors and preferences, which is valuable for targeted marketing strategies and customer retention.

## Features
- Handles missing values and encodes categorical variables.
- Scales numerical data for better clustering performance.
- Reduces data dimensions using PCA for visualization and improved clustering.
- Determines optimal clusters using the elbow method and silhouette score.
- Visualizes clusters using PCA-transformed data.

## Dataset
The project assumes a dataset (`customers.csv`) with customer data, containing numerical and categorical variables. The dataset is processed to handle missing values, encode categorical columns, and scale numerical data.

## Steps
1. **Data Preprocessing**: Handles missing values, encodes categorical variables, and scales numerical features.
2. **PCA**: Reduces dimensionality to 2 principal components for better visualization.
3. **Optimal Clusters**: Determines the best number of clusters using the elbow method and silhouette scores.
4. **K-Means Clustering**: Applies K-Means clustering to group customers into distinct segments.
5. **Cluster Analysis**: Analyzes each cluster to extract insights and patterns.

## Visualizations
- **PCA Scatterplot**: Visual representation of the dataset after PCA.
- **Elbow Method Plot**: Helps identify the optimal number of clusters.
- **Silhouette Score Plot**: Measures cluster quality for different numbers of clusters.
- **Cluster Plot**: Visualizes the K-Means clusters on PCA-reduced data.

## Requirements
Install the dependencies using the following command:
```bash
pip install -r requirements.txt
