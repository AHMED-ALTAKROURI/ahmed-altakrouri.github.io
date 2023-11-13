import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def fit_kmeans_and_evaluate(data, n_clusters=4, n_init=100, max_iter=400, init='k-means++', random_state=42):
    # Make a copy of the data to avoid modifying the original DataFrame
    data_copy = data.copy()

    # Fit KMeans model to the data
    kmeans_model = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, init=init, random_state=random_state)
    kmeans_model.fit(data_copy)

    # Calculate the Silhouette Score of the KMeans model
    silhouette = silhouette_score(data_copy, kmeans_model.labels_, metric='euclidean')
    print('KMeans Scaled Silhouette Score: {}'.format(silhouette))

    # Assign cluster labels to each data point
    labels = kmeans_model.labels_

    # Concatenate the data with the cluster labels into a single DataFrame
    clusters = pd.concat([data_copy, pd.DataFrame({'cluster_scaled': labels})], axis=1)

    return clusters


def visualize_clusters_3d(data, labels_column='cluster_scaled', plot_width=800, plot_height=600):
    # Extract cluster labels
    labels = data[labels_column]

    # Remove cluster labels from data
    data_without_labels = data.drop(labels_column, axis=1)

    # Perform PCA
    pca = PCA(n_components=3)
    components = pca.fit_transform(data_without_labels)

    # Create a DataFrame from PCA results
    pca_df = pd.DataFrame(data=components,
                          columns=['principal component 1', 'principal component 2', 'principal component 3'])

    # Add cluster labels to pca_df
    pca_df[labels_column] = labels

    # Create 3D scatter plot
    fig = px.scatter_3d(pca_df, x='principal component 1', y='principal component 2', z='principal component 3',
                        color=labels_column,
                        # symbol=labels_column,
                        opacity=0.8)

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), width=plot_width, height=plot_height)
    fig.show()


def visualize_clusters_3d_tsne(data, labels_column='cluster_scaled', plot_width=800, plot_height=600):
    # Extract cluster labels
    labels = data[labels_column]

    # Remove cluster labels from data
    data_without_labels = data.drop(labels_column, axis=1)

    # Perform t-SNE
    tsne = TSNE(n_components=3)
    components = tsne.fit_transform(data_without_labels)

    # Create a DataFrame from t-SNE results
    tsne_df = pd.DataFrame(data=components, columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])

    # Add cluster labels to tsne_df
    tsne_df[labels_column] = labels

    # Create 3D scatter plot
    fig = px.scatter_3d(tsne_df, x='Dimension 1', y='Dimension 2', z='Dimension 3',
                        color=labels_column, symbol=labels_column, opacity=0.8)

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), width=plot_width, height=plot_height)
    fig.show()


def find_optimal_dimensions(data):
    # Perform PCA on the data
    pca = PCA()
    pca.fit(data)

    # Calculate the cumulative explained variance ratio
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Plot the cumulative explained variance
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.xlabel('Number of dimensions')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA: Cumulative Explained Variance')
    plt.show()

    # Find the elbow point (optimal number of dimensions)
    diff = np.diff(cumulative_variance)
    elbow_index = np.argmax(diff) + 1

    return elbow_index


def find_optimal_dimensions_tsne(data, perplexity_range):
    # Initialize lists to store dimensions and KL divergence scores
    dims = []
    scores = []

    # Set the maximum number of dimensions based on the input data
    max_dim = min(3, data.shape[1] - 1)

    # Iterate through the range of dimensions
    for dim in range(1, max_dim + 1):
        # Break the loop if the number of dimensions exceeds the length of perplexity_range
        if dim > len(perplexity_range):
            break

        # Perform t-SNE with the current number of dimensions
        tsne = TSNE(n_components=dim)
        embeddings = tsne.fit_transform(data)

        # Append the current dimension and KL divergence score to the lists
        dims.append(dim)
        scores.append(tsne.kl_divergence_)

    # Plot the KL divergence scores
    plt.plot(dims, scores, marker='o')
    plt.xlabel('Number of dimensions')
    plt.ylabel('KL Divergence Score')
    plt.title('t-SNE: KL Divergence')
    plt.show()

    # Find the optimal number of dimensions with the lowest KL divergence score
    optimal_dim_index = scores.index(min(scores))
    optimal_dimensions = dims[optimal_dim_index]

    return optimal_dimensions


def reduce_dimensionality_with_tsne(data, components):
    # Perform t-SNE with the specified number of components
    tsne = TSNE(n_components=components, random_state=423)
    reduced_data = tsne.fit_transform(data)

    # Return the reduced data
    return reduced_data


def reduce_dimensionality_with_pca(data, components):
    # Perform PCA with the specified number of components
    pca = PCA(n_components=components)
    reduced_data = pca.fit_transform(data)

    # Return the reduced data
    return reduced_data
