import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


yelp = pd.read_csv('yelp_data.csv')

yelp_edit = yelp[['permitID', 'rating']]
yelp_edit.to_csv('yelp_edit.csv', index=False)
mapping = pd.read_csv('mapping.csv')


mapping_edit = mapping[['permit_id', 'crit_viol', 'non_crit_viol', 'crit_viol_cos', 'crit_viol_rpt', 'non_crit_viol_cos', 'non_crit_viol_rpt', 'crit_viol_tbr',
       'non_crit_viol_tbr']]
mapping_edit.to_csv('mapping_edit.csv', index=False)

# # working on file yelp_edit and mapping_edit

yelp_df = pd.read_csv('yelp_edit.csv')
mapping_df = pd.read_csv('mapping_edit.csv')

# Merge the datasets on the common column
merged_df_yelp_mapping = pd.merge(yelp_df, mapping_df, left_on='permitID', right_on='permit_id', how='inner')

merged_df_yelp_mapping.to_csv('merged_df_yelp_mapping.csv', index=False)
merged_df_yelp_mapping = pd.read_csv('merged_df_yelp_mapping.csv')

# Aggregations by permitID
agg_df = merged_df_yelp_mapping.groupby('permit_id').agg({
    'rating': 'mean',  
    'crit_viol': 'mean',
    'crit_viol_cos': 'sum',  
    'crit_viol_rpt': 'sum',  
    'crit_viol_tbr': 'sum',  
}).reset_index()
agg_df

# we take permitid seperate from features and will later use it to back to permitid
index_permitid = agg_df[['permit_id']]

agg_df.drop(['permit_id'], axis=1, inplace=True)
X = agg_df

# applying K menas clustering:


## Finding the optimal number of clusters using elbow method

wcss = []

# Try different values of k (e.g., from 1 to 10)
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(agg_df)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(range(1, 15), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.show()

# interperting thhe above we selct k = 4
n_clusters = 4 # You can adjust this based on your domain knowledge or evaluation
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)


# Add cluster labels to our data
X['cluster'] = kmeans.labels_

# we see the following paris of features and select the best which have more spread
feature_pairs = [('rating', 'crit_viol'), ('rating', 'crit_viol_cos'), ('crit_viol', 'crit_viol_cos')]

plt.figure(figsize=(15, 5))
for i, (x_feature, y_feature) in enumerate(feature_pairs, 1):
    plt.subplot(1, 3, i)
    for cluster in range(n_clusters):
        cluster_data = X[X['cluster'] == cluster]
        plt.scatter(cluster_data[x_feature], cluster_data[y_feature], label=f'Cluster {cluster}')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.xlim(0,13)
    plt.ylim(0,21)
    plt.legend()

plt.tight_layout()
plt.show()

# we select the following feature pair to make decision about clusters
feature_pairs = [('crit_viol', 'crit_viol_cos')]

for x_feature, y_feature in feature_pairs:
    plt.figure(figsize=(14, 10))  # Adjust the figsize as needed
    for cluster in range(n_clusters):
        cluster_data = X[X['cluster'] == cluster]
        plt.scatter(cluster_data[x_feature], cluster_data[y_feature], label=f'Cluster {cluster}')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.xlim(0, 13)
    plt.ylim(0, 21)
    plt.legend()
    plt.title(f'Scatter Plot of {x_feature} vs {y_feature}')
    plt.show()

# we can see cluster 2 and cluster 3 have high crit_voil and high crit_voil_cos

cluster_2_indices = X[X['cluster'] == 2].index
cluster_3_indices = X[X['cluster'] == 3].index


cluster_2 = index_permitid.iloc[cluster_2_indices]
print(cluster_2)
cluster_3 = index_permitid.iloc[cluster_3_indices]
print(cluster_3)












