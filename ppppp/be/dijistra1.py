# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import networkx as nx
# import numpy as np
# from scipy.spatial import distance_matrix
# import test
#
#
# def dijkstra(graph, source):
#     n = graph.shape[0]
#     visited = np.zeros(n, dtype=bool)
#     dist = np.inf * np.ones(n)
#     dist[source] = 0
#
#     for _ in range(n):
#         u = np.argmin(dist * (1 - visited))
#         visited[u] = True
#
#         for v in range(n):
#             if not visited[v] and graph[u, v] > 0:
#                 new_dist = dist[u] + graph[u, v]
#                 if new_dist < dist[v]:
#                     dist[v] = new_dist
#
#     return dist
#
# df = pd.read_csv(r"E:\B.Tech sem 4\MP & MC\ppppp\be\income.csv")
#
# km = KMeans(n_clusters=3)
# y_predicted = km.fit_predict(df[["Latitude", "Longitude"]])
#
# df['cluster'] = y_predicted
#
# df1 = df[df.cluster == 0]
# df2 = df[df.cluster == 1]
# df3 = df[df.cluster == 2]
#
#
# plt.figure(figsize=(8, 6))
# plt.scatter(df1["Latitude"], df1['Longitude'], color='green', label='Cluster 0')
# plt.scatter(df2["Latitude"], df2['Longitude'], color='red', label='Cluster 1')
# plt.scatter(df3["Latitude"], df3['Longitude'], color='blue', label='Cluster 2')
# plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='Centroid')
# plt.legend()
# plt.xlabel('Latitude')
# plt.ylabel('Longitude')
# plt.title('K-means Clustering Graph')
#
# data = df[["Latitude", "Longitude"]].values
# distances = distance_matrix(data, data)  # Compute pairwise distances
#
# source_latitude = float(input("Enter the source latitude :"))
# source_longitude = float(input("Enter the source longitude : "))
#
# source_vertex = np.argmin(np.sqrt(np.power(data[:, 0] - source_latitude, 2) + np.power(data[:, 1] - source_longitude, 2)))
#
# shortest_paths = dijkstra(distances, source_vertex)
#
# dijkstra_graph = nx.from_numpy_array(distances)
#
# node_colors = ['yellow' if i == source_vertex else 'green' if shortest_paths[i] < np.inf else 'gray' for i in range(len(data))]
#
# centroid_indices = km.labels_
# centroid_distances = []
# shortest_paths_to_centroids = []
#
# for centroid_index in np.unique(centroid_indices):
#     centroid_distances.append(shortest_paths[np.where(centroid_indices == centroid_index)].min())
#     shortest_path_indices = np.argmin(distances[np.where(centroid_indices == centroid_index), :], axis=1)
#     shortest_paths_to_centroids.append(shortest_path_indices)
#
# centroid_nodes = np.unique(np.concatenate(shortest_paths_to_centroids))
# centroid_dijkstra_graph = dijkstra_graph.subgraph(centroid_nodes)
#
# centroid_node_colors = ['yellow' if i == source_vertex else 'green' if shortest_paths[i] < np.inf else 'gray' for i in centroid_nodes]
#
# shortest_path_edges = []
# for shortest_path_indices in shortest_paths_to_centroids:
#     shortest_path_edges.extend([(shortest_path_indices[i], shortest_path_indices[i+1]) for i in range(len(shortest_path_indices)-1)])
#
# plt.figure(figsize=(8, 6))
# nx.draw_networkx(centroid_dijkstra_graph, pos=data[centroid_nodes], node_color=centroid_node_colors, with_labels=False, node_size=150, width=0.2)
# nx.draw_networkx_edges(centroid_dijkstra_graph, pos=data[centroid_nodes], edgelist=shortest_path_edges, edge_color='orange', width=2)
# plt.xlabel('Latitude')
# plt.ylabel('Longitude')
# plt.title('Dijkstra Graph for Centroids')
#
# closest_centroid_index = np.argmin(centroid_distances)
# closest_centroid = km.cluster_centers_[closest_centroid_index]
#
# print("Closest Centroid to the Source:")
# print(closest_centroid)
#
# centroids_df = pd.DataFrame(km.cluster_centers_, columns=["Centroid Latitudes", "Centroid Longitude"])
#
# print("Centroids:")
# print(centroids_df)
#
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix


class ClusteringGraph:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.km = None
        self.distances = None
        self.data = None
        self.source_vertex = None
        self.shortest_paths = None
        self.dijkstra_graph = None
        self.node_colors = None
        self.centroid_indices = None
        self.centroid_distances = None
        self.shortest_paths_to_centroids = None
        self.centroid_nodes = None
        self.centroid_dijkstra_graph = None
        self.centroid_node_colors = None
        self.shortest_path_edges = None
        self.closest_centroid = None
        self.centroids_df = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)

    def perform_clustering(self, n_clusters):
        self.km = KMeans(n_clusters=n_clusters, n_init=10)
        y_predicted = self.km.fit_predict(self.df[["Latitude", "Longitude"]])
        self.df['cluster'] = y_predicted

    def plot_clusters(self):
        df_clusters = []
        for i in range(self.km.n_clusters):
            df_clusters.append(self.df[self.df.cluster == i])

        plt.figure(figsize=(8, 6))
        for i, df_cluster in enumerate(df_clusters):
            plt.scatter(df_cluster["Latitude"], df_cluster['Longitude'], label='Cluster {}'.format(i))
        plt.scatter(self.km.cluster_centers_[:, 0], self.km.cluster_centers_[:, 1], color='black', marker='*',
                    label='Centroids')
        plt.legend()
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.title('K-means Clustering Graph')

    def compute_distances(self):
        self.data = self.df[["Latitude", "Longitude"]].values
        self.distances = distance_matrix(self.data, self.data)

    def set_source_vertex(self, latitude, longitude):
        source = np.argmin(np.sqrt(np.power(self.data[:, 0] - latitude, 2) + np.power(self.data[:, 1] - longitude, 2)))
        self.source_vertex = source

    def compute_shortest_paths(self):
        n = len(self.data)
        visited = np.zeros(n, dtype=bool)
        dist = np.inf * np.ones(n)
        dist[self.source_vertex] = 0

        for _ in range(n):
            u = np.argmin(dist * (1 - visited))
            visited[u] = True

            for v in range(n):
                if not visited[v] and self.distances[u, v] > 0:
                    new_dist = dist[u] + self.distances[u, v]
                    if new_dist < dist[v]:
                        dist[v] = new_dist

        self.shortest_paths = dist
        self.dijkstra_graph = nx.from_numpy_array(self.distances)

    def plot_dijkstra_graph(self):
        node_colors = ['yellow' if i == self.source_vertex else 'green' if self.shortest_paths[i] < np.inf else 'gray'
                       for i in range(len(self.data))]

        centroid_indices = self.km.labels_
        centroid_distances = []
        shortest_paths_to_centroids = []

        for centroid_index in np.unique(centroid_indices):
            centroid_distances.append(self.shortest_paths[np.where(centroid_indices == centroid_index)].min())
            shortest_path_indices = np.argmin(self.distances[np.where(centroid_indices == centroid_index), :], axis=1)
            shortest_paths_to_centroids.append(shortest_path_indices)

        centroid_nodes = np.unique(np.concatenate(shortest_paths_to_centroids))
        self.centroid_nodes = centroid_nodes
        self.centroid_dijkstra_graph = self.dijkstra_graph.subgraph(centroid_nodes)

        centroid_node_colors = [
            'yellow' if i == self.source_vertex else 'green' if self.shortest_paths[i] < np.inf else 'gray' for i in
            centroid_nodes]
        self.centroid_node_colors = centroid_node_colors

        shortest_path_edges = []
        for shortest_path_indices in shortest_paths_to_centroids:
            shortest_path_edges.extend([(shortest_path_indices[i], shortest_path_indices[i + 1]) for i in
                                        range(len(shortest_path_indices) - 1)])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        nx.draw_networkx(self.centroid_dijkstra_graph, pos=self.data[centroid_nodes],
                         node_color=self.centroid_node_colors, with_labels=False, node_size=150, width=0.2, ax=ax1)
        nx.draw_networkx_edges(self.centroid_dijkstra_graph, pos=self.data[centroid_nodes],
                               edgelist=shortest_path_edges, edge_color='orange', width=2, ax=ax1)
        ax1.set_xlabel('Latitude')
        ax1.set_ylabel('Longitude')
        ax1.set_title('Dijkstra Graph for Centroids')

        ax2.scatter(self.df["Latitude"], self.df['Longitude'], c=node_colors, label='Nodes')
        ax2.scatter(self.km.cluster_centers_[:, 0], self.km.cluster_centers_[:, 1], color='black', marker='*',
                    label='Centroids')
        ax2.legend()
        ax2.set_xlabel('Latitude')
        ax2.set_ylabel('Longitude')
        ax2.set_title('K-means Clustering Graph with Dijkstra Paths')

        plt.tight_layout()
        #plt.show()
        plt.savefig("graph.png")

    def find_closest_centroid(self):
        centroid_indices = self.km.labels_
        centroid_distances = []
        shortest_paths_to_centroids = []

        for centroid_index in np.unique(centroid_indices):
            centroid_distances.append(self.shortest_paths[np.where(centroid_indices == centroid_index)].min())

        closest_centroid_index = np.argmin(centroid_distances)
        closest_centroid = self.km.cluster_centers_[closest_centroid_index]
        self.closest_centroid = closest_centroid

    def print_closest_centroid(self):
        print("Closest Centroid to the Source:")
        print(self.closest_centroid)

    def compute_centroids(self):
        centroids_df = pd.DataFrame(self.km.cluster_centers_, columns=["Centroid Latitudes", "Centroid Longitude"])
        self.centroids_df = centroids_df

    def print_centroids(self):
        print("Centroids:")
        print(self.centroids_df)


# Usage example
data_path = r"C:\Users\vicky\Desktop\IV SEM PROJ\info.csv"

graph = ClusteringGraph(data_path)
graph.load_data()
graph.perform_clustering(n_clusters=3)
graph.plot_clusters()
graph.compute_distances()
graph.set_source_vertex(latitude=12.345, longitude=67.890)  # Replace with actual latitude and longitude
graph.compute_shortest_paths()
graph.plot_dijkstra_graph()
graph.find_closest_centroid()
graph.print_closest_centroid()
graph.compute_centroids()
graph.print_centroids()

