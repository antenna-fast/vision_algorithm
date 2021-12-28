from numpy import *
from sklearn.cluster import DBSCAN


# 聚类

# Compute DBSCAN
db = DBSCAN(eps=1, min_samples=3).fit(key_pts_buff_1)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('labels:', labels)
unique_labels = set(labels)  # 列表变成集合
print(unique_labels)
