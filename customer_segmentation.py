import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- STEP 0: Generate Dummy Data (Since we don't have the CSV) ---
# Creating a dataset similar to the structure shown in the instructions
data = {
    'CustomerID': range(1, 201),
    'Gender': np.random.choice(['Male', 'Female'], 200),
    'Age': np.random.randint(18, 70, 200),
    'Annual Income (k$)': np.random.randint(15, 140, 200),
    'Spending Score (1-100)': np.random.randint(1, 100, 200)
}
df = pd.DataFrame(data)
df.to_csv('retail_data.csv', index=False)
print("Step 0: 'retail_data.csv' created successfully.\n")

# --- STEP 1: Load the Data  ---
df = pd.read_csv('retail_data.csv')
print("First 5 rows of the dataset:")
print(df.head())

# --- STEP 2: Preprocessing [cite: 265] ---
# We will use 'Annual Income' and 'Spending Score' for clustering
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Standardize the features 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- STEP 3: K-Means Clustering (The Elbow Method) [cite: 271] ---
# Determining optimal clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# --- STEP 4: Apply K-Means  ---
# Based on the elbow graph, we usually choose the "bend". Let's assume 5 clusters.
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to original data [cite: 299]
df['Cluster'] = clusters

# --- STEP 5: Visualize the Clusters  ---
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=df['Annual Income (k$)'], 
    y=df['Spending Score (1-100)'], 
    hue=df['Cluster'], 
    palette='viridis', 
    s=100
)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

# --- STEP 6: Analyze the Clusters [cite: 317] ---
print("\nCluster Analysis (Mean values):")
print(df.groupby('Cluster')[features].mean())