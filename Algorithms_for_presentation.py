from sklearn.cluster import HDBSCAN, DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from DBCV import DBCV as dbcv
import numpy as np
import matplotlib.pyplot as plt



def main():
    data_points = load_data()
    label_removed_list = [(x, y) for x, y, label in data_points]

    #scuffed method to only get the desired algorithms to run
    #hdbscan(label_removed_list, data_points)
#     for i in [1,1.5,1.75,2,2.5,3]:
    #    for j in range(1,5):
#         dbscan(label_removed_list, data_points,eps=i,min_samples=j,label_header=f"DBSCAN Clustering for eps={i}, min_points={j}")
    for i in [1,2,3,4,5,6,7,8,9]:
        kmeans(data_points,label_removed_list=label_removed_list,cluster_count=i,
               file_name=f"KMeans_Clustering_temp{i}_clusters")
    #spectral_clustering(data_points, label_removed_list, cluster_count=2)

def hdbscan(label_removed_list, data_points):
    #algorithm parameter is simply the way to calculate the nearest neighbours, where k-d tree and ball tree are just tree based algorithms to find the nearest neighbours.
    hdbscanner = HDBSCAN(algorithm='auto',metric='euclidean', min_cluster_size=5) 
    results_hdbscan = hdbscanner.fit_predict(label_removed_list)
    visualize_data(data_points, results_hdbscan, display_plot=True, display_dbcv=True,display_silhouette_score=True,display_centroids=False,
                    save=False,file_name="Hdbscan"+".png",plot_header="HDBSCAN Clustering")

def dbscan(label_removed_list,data_points, eps, min_samples,label_header=None):
    dbscanner = DBSCAN(eps=eps,min_samples=min_samples,metric='euclidean')
    results_dbscan = dbscanner.fit_predict(label_removed_list)
    print(f"{calculate_silhouette_score(label_removed_list, results_dbscan)}")

    visualize_data(data_points, results_dbscan, display_plot=False, save=True,
                   display_dbcv=True,display_silhouette_score=True,display_centroids=False,
                   plot_header=label_header,file_name = label_header+"temp.png")

def kmeans(data_points,label_removed_list,cluster_count,file_name=None):
    kmeans = KMeans(n_clusters=cluster_count)
    kmeans.fit(label_removed_list)
    results = kmeans.labels_
    centroids = kmeans.cluster_centers_


    total_clusters = cluster_count

    # Generate sufficient colors
    colors = plt.cm.get_cmap('tab10', total_clusters)
    cluster_colors = {i: colors(i) for i in range(total_clusters)}
    
    for i, (x, y, _) in enumerate(data_points):
        cluster_label = int(results[i])
        color = cluster_colors.get(cluster_label, 'black')
        plt.scatter(x, y, color=color)

    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', s=200, label='Centroids')
    if len(set(results)) > 1:
        silhouette_score = calculate_silhouette_score(label_removed_list, results)
        plt.text(0.01, 0.99, f"Silhouette Score: {silhouette_score}", fontsize=10, ha='left', va='top', transform=plt.gcf().transFigure)
    
    plt.title(file_name)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend(loc='best', fontsize='small', markerscale=0.5)
    #plt.savefig(file_name)
    plt.clf()  # Clear the current figure to avoid overlapping plots


def spectral_clustering(data_points, label_removed_list, cluster_count):
    spectral = SpectralClustering(n_clusters=cluster_count, affinity='nearest_neighbors', n_neighbors=5)
    spectral.fit(label_removed_list)
    results = spectral.labels_

    visualize_data(data_points, results, display_plot=True, save=True,plot_header="Spectral Clustering",
                   display_centroids=False, display_silhouette_score=True, display_dbcv=True,
                   file_name="Spectral_Clustering.png")

def visualize_data(data_points, result_from_scan,plot_header,display_plot:bool, save:bool,
                   display_silhouette_score, display_centroids, display_dbcv,
                   file_name=None):
     #find the colours to generate
    total_clusters = max(result_from_scan) + 1
    plt.clf()  # Clear the current figure to avoid overlapping plots

    # Generate sufficient colors
    colors = plt.cm.get_cmap('tab10', total_clusters)
    cluster_colors = {i: colors(i) for i in range(total_clusters)}
    label_removed_points = [(x, y) for x, y, label in data_points]
    for i, (x, y, _) in enumerate(data_points):
        cluster_label = int(result_from_scan[i])
        color = cluster_colors.get(cluster_label, 'black')
        plt.scatter(x, y, color=color)


    plt.title(plot_header)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend(loc='best', fontsize='small', markerscale=0.5)

    if display_silhouette_score and len(set(result_from_scan)) > 1:
        silhouette_score = calculate_silhouette_score(label_removed_points, result_from_scan)
        plt.text(0.01, 0.99, f"silhouette score: {silhouette_score}", fontsize=10, ha='left', va='top', transform=plt.gcf().transFigure)
 
    if display_centroids:
        centroids = calculate_centroids(data_points, result_from_scan, total_clusters)
        for i, centroid in centroids.items():
            plt.scatter(centroid[0], centroid[1], color='red', marker='X', s=200, label=f'Centroid {i}')

    if display_dbcv:
        #note this makes noise a cluster too.
        #Results don't change the outcome when these are filtered out. Considerable for other examples.
        dbcv_result = dbcv(data_points, result_from_scan)

        plt.text(0.01, 0.95, f"DBCV: {dbcv_result}", fontsize=10, ha='left', va='top', transform=plt.gcf().transFigure)
    if save and file_name is not None:
        plt.savefig(file_name)

    if display_plot:
        plt.show()
 
def load_data():
    datapoints = []
    with open("dataset.txt", "r") as f:
        for line in f:
            line = line.strip()
            x, y, label = line.split()
            datapoints.append((float(x), float(y), label))
    return datapoints

def generate_cluster_dict(data_points,cluster, total_clusters):
    cluster_dict = {}
    for i in range(total_clusters):
        cluster_dict[i] = []
    for i in range(len(data_points)):
        point = data_points[i]
        cluster_label = cluster[i]
        cluster_dict[int(cluster_label)].append(point)

    return cluster_dict




def calculate_silhouette_score(label_removed_list, results):
    labels = np.array(results)
    score = silhouette_score(label_removed_list, labels=labels)
    return score

def calculate_centroids(data_points, results, total_clusters):
    total_clusters = max(results) + 1
    cluster_dict = generate_cluster_dict(data_points,results, total_clusters)
    centroids = {}
    for i in range(total_clusters):
        points = cluster_dict[i]
        if len(points) > 0:
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            centroids[i] = (np.mean(x_coords), np.mean(y_coords))
    return centroids


if __name__ == "__main__":
    main()