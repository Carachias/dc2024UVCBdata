"""This file contains all the relevant and reusable functions for the clustering functionality"""

# Library imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from itertools import permutations


def remap_labels(pred_labels, true_labels):
    """This Function renames the clusters of the predicted labels in a way that fits best to the true labels.

    Algorithmically this approach is pretty naive as all permutations of the class names are tried until the best one
    is found. Because of that the function becomes incredibly slow with more than 7 classes and the
    relabeling is skipped if there are more than 7 classes and the original labels are returned as they are.
    :param pred_labels: list of labels predicted by a clustering method
    :param true_labels: actual class labels
    :return:
    """
    # if there are more than 7 classes skip the relabeling due to function inefficiency
    if len(np.unique(pred_labels)) > 7:
        return pred_labels

    # if there are 7 or fewer classes start the relabeling process
    best_relabel = pred_labels
    cluster_labels = np.unique(best_relabel)
    cluster_label_permutations = list(permutations(cluster_labels))

    max_inters = 0
    for permutation in cluster_label_permutations:
        remap = np.choose(pred_labels, permutation)
        intersects = sum(a == b for a, b in zip(remap, true_labels))
        if intersects > max_inters:
            best_relabel = remap
            max_inters = intersects

    return best_relabel


def combine_dataframes(ground_truth, toxpi):
    """Combine the data of two dataframes/ tables into one.

    The resulting dataframe can be passed to other functions for further processing.
    :param ground_truth: dataframe containing a (sub)set of ground truth shaped datapoints
    :param toxpi: dataframe containing ToxPi datapoints
    :return:
    """
    combined_dataframe = ground_truth.merge(toxpi, how='inner', left_on="Substance", right_on="Name")

    return combined_dataframe


def perform_dataframe_preprocessing(dataframe, data_origin):
    """Perform basic preprocessing of the input dataframe.

    Remove undesired features and rows/ datapoints that carry no information (zero-rows).
    Extract class labels from remaining rows. Return preprocessed dataframe and extracted labels.
    :param dataframe: input dataframe that has to be preprocessed
    :param data_origin: string specifying if data is of shape ground truth, ToxPi or a combination
    :return:
    """
    # remove the zero rows in case it has not been done before
    no_zero_rows_dataframe = dataframe
    no_zero_rows_dataframe.drop(no_zero_rows_dataframe.loc[no_zero_rows_dataframe['mg'] == 0].index, inplace=True)

    # extract labels of the remaining rows
    no_zero_rows_class_labelframe = no_zero_rows_dataframe[['Class']]

    # turn labels into ints. these can later be used to color the datapoints in plots
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(no_zero_rows_class_labelframe.Class)
    no_zero_rows_class_labels_as_colors = label_encoder.transform(no_zero_rows_class_labelframe.Class)

    # remove undesired features (non-numerical and class labels for example)
    if data_origin == "GT only":
        no_zero_rows_dataframe.drop(labels=["Substance", "Class", "Sample.ID", "total.Wt.Percent", "mg"], axis=1, inplace=True)
    if data_origin == "ToxPi only":
        no_zero_rows_dataframe.drop(labels=["ToxPi Score", "HClust Group", "KMeans Group", "Name", "Source",
                                            "Substance", "Class", "Sample.ID", "total.Wt.Percent", "mg",
                                            "Per1", "Per2", "Per3", "Per4", "Per5", "Per6", "Per7",
                                            "P3_7_index", "P4_7_index", "P5_7_index", "P1_2_index",
                                            "P1_index", "P2_index", "P3_index", "P4_index", "P5_index", "P6_index",
                                            "P7_index", "Primary_PAH"], axis=1, inplace=True)
    if data_origin == "Combination":
        no_zero_rows_dataframe.drop(labels=["ToxPi Score", "HClust Group", "KMeans Group", "Name", "Source",
                                            "Substance", "Class", "Sample.ID", "total.Wt.Percent", "mg"], axis=1, inplace=True)

    return no_zero_rows_dataframe, no_zero_rows_class_labelframe, no_zero_rows_class_labels_as_colors


def plot_clustering_result(data, labels, plot_title):
    """Use the given labels to color the datapoints and return the colored plot.

    :param data: coordinates of the datapoints after dimensionality reduction
    :param labels: integer-converted labels to color the datapoints with
    :param plot_title: string that is used as the title for the plot
    :return:
    """
    fig, ax = plt.subplots()
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.title(plot_title)
    fig
    return fig


def calc_amount_correct_labels(labels, truth_vector):
    """Compare clustering results with actual ground_truth labels.

    Most useful for if there are only two classes in the data, essentially representing binary classification.
    :param labels: labels returned by a clustering algorithm
    :param truth_vector: actual labels
    :return:
    """
    num_correct_labels = 0
    loopctr = 0
    for each in labels:
        if each == truth_vector[loopctr]:
            num_correct_labels += 1
        else:
            pass
        loopctr += 1

    correct_labels_percent = (num_correct_labels / len(truth_vector)) * 100

    return num_correct_labels, correct_labels_percent


def pca_dimensionality_reduction(dataframe, num_components):
    """Reduce dimensionality of the input dataframe with basic PCA.

    :param dataframe: input dataframe to reduce dimensionality of
    :param num_components: amount of components to use for PCA (given by a slider)
    :return:
    """
    pca = PCA(n_components=num_components)
    pca_results = pca.fit_transform(dataframe)
    return pca_results


def tsne_dimensionality_reduction(dataframe, num_components, perplexity, iters):
    """Reduce dimensionality with TSNE.

    :param dataframe: input dataframe to reduce dimensionality of
    :param num_components: amount of components to use for tSNE (given by a slider)
    :param perplexity: perplexity value (obtained from slider)
    :param iters: number of iterations (obtained from slider)
    :return:
    """
    tsne = TSNE(n_components=num_components, verbose=1, perplexity=perplexity, max_iter=iters)
    tsne_results = tsne.fit_transform(dataframe)
    return tsne_results


def perform_kmeans_clustering(data, num_clusters, random_state, truth_vector):
    """Perform kmeans clustering on data that has been transformed by some dimensionality reduction method.

    Generate and return two plots: One of the data colored with the labels found through kmeans and a second one
    colored with the actual ground truth labels.
    Calculate silhouette coefficients for both plots and return them as well.
    :param data: input data, possibly after applying dimensionality reduction
    :param num_clusters: number of classes/ clusters in the input data
    :param random_state: integer necessary to get deterministic results
    :param truth_vector: actual labels for the datapoints extracted from the ground truth table
    :return:
    """
    # transform (standardize) the input data
    standardized_data = StandardScaler().fit_transform(data)

    # perform kmeans on the standardized data
    kmeans_result = KMeans(n_clusters=num_clusters, random_state=random_state).fit(standardized_data)
    labels = kmeans_result.labels_

    # try to find a label match
    remapped_labels = remap_labels(labels, truth_vector)

    # calculate silhouette coefficients for the kmeans clusters vs. the actual clusters
    kmeans_sil_co = metrics.silhouette_score(standardized_data, labels)
    true_kmeans_sil_co = metrics.silhouette_score(standardized_data, truth_vector)

    # plot the datapoints with the labels found by kmeans
    kmeans_colored_plot = plot_clustering_result(standardized_data, remapped_labels, 'standardized_kmeans')
    # plot the same datapoints with the actual labels
    correctly_colored_plot = plot_clustering_result(standardized_data, truth_vector, 'true_standardized_kmeans')

    # calculate the amount of correct labels
    num_correct_labels, correct_labels_percent = calc_amount_correct_labels(remapped_labels, truth_vector)

    return kmeans_colored_plot, correctly_colored_plot, kmeans_sil_co, true_kmeans_sil_co, num_correct_labels, correct_labels_percent


def perform_spectral_clustering(data, num_clusters, num_components, random_state, truth_vector):
    """Perform spectral clustering on data that has been transformed by some dimensionality reduction method.

    Generate and return two plots: One of the data colored with the labels found through spectral clustering
    and a second one colored with the actual ground truth labels.
    Calculate silhouette coefficients for both plots and return them as well.
    :param data: input data, possibly after applying dimensionality reduction
    :param num_clusters: number of classes/ clusters in the input data
    :param num_components: number of eigenvectors to use for the spectral embedding
    :param random_state: integer necessary to get deterministic results
    :param truth_vector: actual labels for the datapoints extracted from the ground truth table
    :return:
    """
    # transform (standardize) the input data
    standardized_data = StandardScaler().fit_transform(data)

    # perform spectral clustering on the standardized data
    spectral_result = SpectralClustering(n_clusters=num_clusters, n_components=num_components, assign_labels='kmeans', affinity='nearest_neighbors', random_state=random_state).fit(standardized_data)
    labels = spectral_result.labels_

    # try to find a label match
    remapped_labels = remap_labels(labels, truth_vector)

    # calculate silhouette coefficients for the spectral clusters vs. the actual clusters
    spectral_sil_co = metrics.silhouette_score(standardized_data, remapped_labels)
    true_spectral_sil_co = metrics.silhouette_score(standardized_data, truth_vector)

    # plot the datapoints with the labels found through spectral clustering
    spectral_colored_plot = plot_clustering_result(standardized_data, remapped_labels, 'standardized_spectral')
    # plot the same datapoints with the actual labels
    correctly_colored_plot = plot_clustering_result(standardized_data, truth_vector, 'true_standardized_spectral')

    # calculate the amount of correct labels
    num_correct_labels, correct_labels_percent = calc_amount_correct_labels(remapped_labels, truth_vector)

    return spectral_colored_plot, correctly_colored_plot, spectral_sil_co, true_spectral_sil_co, num_correct_labels, correct_labels_percent


def perform_dbscan_clustering(data, epsilon, min_samples, truth_vector):
    """Perform dbscan clustering on data that has been transformed by some dimensionality reduction method.

    Generate and return two plots: One of the data colored with the labels found through dbscan clustering
    and a second one colored with the actual ground truth labels.
    Calculate silhouette coefficients for both plots and return them as well.
    :param data: input data, possibly after applying dimensionality reduction
    :param epsilon: epsilon value
    :param min_samples: number of samples per point
    :param truth_vector: actual labels for the datapoints extracted from the ground truth table
    :return:
    """
    # transform (standardize) the input data
    standardized_data = StandardScaler().fit_transform(data)

    # perform dbscan clustering on the standardized data
    dbscan_result = DBSCAN(eps=(epsilon/10), min_samples=min_samples).fit(standardized_data)
    labels = dbscan_result.labels_

    # find out if at least one non-noise cluster has been found
    n_clusters = len(np.unique(labels))

    # if exactly one non-noise cluster is found, convert the labels from [-1, 0] to [0, 1]
    converted_labels = []
    for each in labels:
        converted_labels.append(int(each) + 1)

    # try to find a label match
    remapped_labels = remap_labels(converted_labels, truth_vector)

    # calculate silhouette coefficients for the dbscan clusters vs. the actual clusters
    if n_clusters >= 1:
        dbscan_sil_co = metrics.silhouette_score(standardized_data, remapped_labels)
    else:
        dbscan_sil_co = 'not enough clusters'
    true_dbscan_sil_co = metrics.silhouette_score(standardized_data, truth_vector)

    # plot the datapoints with the labels found through dbscan clustering
    dbscan_colored_plot = plot_clustering_result(standardized_data, remapped_labels, 'standardized_dbscan')
    # plot the same datapoints with the actual labels
    correctly_colored_plot = plot_clustering_result(standardized_data, truth_vector, 'true_standardized_dbscan')

    # calculate the amount of correct labels
    num_correct_labels, correct_labels_percent = calc_amount_correct_labels(remapped_labels, truth_vector)

    return dbscan_colored_plot, correctly_colored_plot, dbscan_sil_co, true_dbscan_sil_co, num_correct_labels, correct_labels_percent
