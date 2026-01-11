import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

df = pd.read_csv("diamonds_clean_final.csv")

cluster_features = [
    'stone.carat',
    'stone.depth',
    'stone.tableSize',
    'enriched.ratio',
    'enriched.pricePerCarat'
]

X_cluster = df[cluster_features].dropna()

print("Размер данных для кластеризации:", X_cluster.shape)

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Первая проверка — метод локтя (Elbow Method)
inertia = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K, inertia, marker='o')
plt.xlabel('Число кластеров (k)')
plt.ylabel('Inertia')
plt.title('Определение оптимального числа кластеров методом локтя')
plt.grid(True)
plt.show()


# Построение модели K-means с оптимальным числом кластеров
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Добавляем метки кластеров в данные
X_cluster = X_cluster.copy()
X_cluster['cluster'] = clusters

print(X_cluster['cluster'].value_counts())

# Анализ центра кластеров
# Центры кластеров в исходном масштабе признаков
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=cluster_features
)

cluster_centers['cluster'] = cluster_centers.index
print(cluster_centers.to_string(index=False))


# Оценка качества кластеризации

# Silhouette Score

sil_score = silhouette_score(X_scaled, clusters)
print("Silhouette score:", round(sil_score, 3))

dbi = davies_bouldin_score(X_scaled, clusters)
ch = calinski_harabasz_score(X_scaled, clusters)

print("Davies-Bouldin Index:", round(dbi, 3))
print("Calinski-Harabasz Index:", round(ch, 1))



# DBSCAN
from sklearn.cluster import DBSCAN


# Подбор оптимального eps
from sklearn.neighbors import NearestNeighbors

# число ближайших соседей
k = 10

neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# берём расстояние до k-го соседа
k_distances = np.sort(distances[:, k-1])

plt.figure(figsize=(6, 4))
plt.plot(k_distances)
plt.ylim(0, 2.5)
plt.xlabel('Точки (отсортированные)')
plt.ylabel(f'{k}-distance')
plt.title('k-distance plot для подбора eps')
plt.grid(True)
plt.show()

eps_list = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

for eps in eps_list:
    db = DBSCAN(eps=eps, min_samples=10)
    labels = db.fit_predict(X_scaled)

    n_noise = (labels == -1).sum()
    noise_share = n_noise / len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # silhouette только по не-шуму
    from sklearn.metrics import silhouette_score
    mask = labels != -1
    sil = None
    if len(set(labels[mask])) > 1:
        sil = silhouette_score(X_scaled[mask], labels[mask])

    print(f"eps={eps:.2f} | clusters={n_clusters} | noise={noise_share:.3f} | sil={None if sil is None else round(sil,3)}")


dbscan = DBSCAN(
    eps=0.75,      # подобрали по k-distance plot
    min_samples=10
)

clusters_dbscan = dbscan.fit_predict(X_scaled)

# Добавляем метки
X_cluster_dbscan = X_cluster.copy()
X_cluster_dbscan['cluster_dbscan'] = clusters_dbscan

print(X_cluster_dbscan['cluster_dbscan'].value_counts())


# Считаем silhouette_score без шума
mask = clusters_dbscan != -1

if len(set(clusters_dbscan[mask])) > 1:
    sil_dbscan = silhouette_score(
        X_scaled[mask],
        clusters_dbscan[mask]
    )
    print("Silhouette score (DBSCAN):", round(sil_dbscan, 3))
else:
    print("Недостаточно кластеров для оценки")


df_clustered = X_cluster.copy()
df_clustered['cluster_dbscan'] = clusters_dbscan

# средние значения признаков по кластерам DBSCAN
print(df_clustered.groupby('cluster_dbscan')[cluster_features].mean().round(3).to_string())


# Пробуем метод HDBSCAN
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=30,
    min_samples=10
)

labels_hdb = clusterer.fit_predict(X_scaled)

# добавляем метки к данным
X_cluster_hdb = X_cluster.copy()
X_cluster_hdb['cluster_hdb'] = labels_hdb

print(X_cluster_hdb['cluster_hdb'].value_counts())

# Считаем silhouette_score без шума
mask = labels_hdb != -1

if len(set(labels_hdb[mask])) > 1:
    sil_hdb = silhouette_score(
        X_scaled[mask],
        labels_hdb[mask]
    )
    print("Silhouette score (HDBSCAN):", round(sil_hdb, 3))
else:
    print("Недостаточно кластеров для оценки")

# Доля шума
noise_share = (labels_hdb == -1).sum() / len(labels_hdb)
print("Доля шума:", round(noise_share, 3))

# # Иерархическая кластеризация
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import silhouette_score
#
# X = df[cluster_features].dropna()
# Xs = StandardScaler().fit_transform(X)
#
# # # Подбор числа кластеров k
# # for k in range(2, 8):
# #     model = AgglomerativeClustering(
# #         n_clusters=k,
# #         linkage='ward'
# #     )
# #     labels = model.fit_predict(Xs)
# #
# #     sil = silhouette_score(Xs, labels)
# #     print(f"k = {k}, Silhouette = {sil:.3f}")
#
#
# agg = AgglomerativeClustering(
#     n_clusters=4,
#     linkage='ward'
# )
#
# clusters_agg = agg.fit_predict(Xs)
#
# X_cluster_agg = X.copy()
# X_cluster_agg['cluster_agg'] = clusters_agg
#
# # Кол-во значений по кластерам
# counts = X_cluster_agg['cluster_agg'].value_counts()
# percentages = counts / counts.sum() * 100
#
# cluster_stats = pd.DataFrame({
#     'count': counts,
#     'percent': percentages.round(2)
# })
#
# print(cluster_stats)
#
#
#
#
# cluster_profile_agg = (
#     X_cluster_agg
#     .groupby('cluster_agg')[cluster_features]
#     .mean()
#     .round(3)
# )
#
# print(cluster_profile_agg.to_string())

