"""This file contains the frontend for the clustering functionality of the app."""

# Library imports
import streamlit as st
import pandas as pd

# File imports
import backend.clusterer_backend as cl_be


# Draw a title and some text to the app:
st.title('''Dataset Clustering''')
st.header('''Upload some Data:''')
st.subheader('''Ground-Truth:''')
st.markdown('''To use this part of the app, you have to upload any dataset that comes in the shape
of the "Ground Truth"-data, meaning you can either upload the full dataset provided by the paper
or a custom subset that was created using the data modification tool provided by the app:''')
upload_gt_file = st.file_uploader(label="Upload 'Ground Truth'-like data:", type='csv', accept_multiple_files=False)

if upload_gt_file is None:
    gt_dataframe = pd.read_csv("data/modified/two_2_largest_classes.csv")

if upload_gt_file is not None:
    gt_dataframe = pd.read_csv(upload_gt_file)

option_include_toxpi = st.selectbox("What kind of input data do you want to use?", ("ToxPi only", "GT only", "Combination"))
st.write("You selected:", option_include_toxpi)

# remove undesired features as well as useless datapoints (Zero-Rows) in case data modification tool was not used
if option_include_toxpi in ["ToxPi only", "Combination"]:
    toxpi_dataframe = pd.read_csv("data/qcpheno.out_results.csv")
    combined_dataframe = cl_be.combine_dataframes(gt_dataframe, toxpi_dataframe)
    combined_dataframe
    st.write(option_include_toxpi)
    gt_dataframe_preprocessed, gt_preprocessed_labels, gt_preprocessed_labels_as_colors = cl_be.perform_dataframe_preprocessing(combined_dataframe, option_include_toxpi)
if option_include_toxpi == "GT only":
    gt_dataframe_preprocessed, gt_preprocessed_labels, gt_preprocessed_labels_as_colors = cl_be.perform_dataframe_preprocessing(gt_dataframe, option_include_toxpi)

number_datapoints = gt_preprocessed_labels.shape[0]

st.header('''Input Data after Preprocessing:''')
col0, col1, col2, = st.columns([9, 3, 2])
with col0:
    # st.subheader('''Data (without Labels):''')
    st.markdown('''Removed undesired features and useless Datapoints:''')
    gt_dataframe_preprocessed
with col1:
    # st.subheader('''Labels:''')
    st.markdown('''String-Labels:''')
    st.dataframe(gt_preprocessed_labels, width=130)
with col2:
    st.markdown('''Int-Labels:''')
    st.dataframe(gt_preprocessed_labels_as_colors, width=100)

st.markdown('''***''')

# add ui elements to change settings for dimensionality reduction using PCA
st.header('''Dimensionality reduction with PCA:''')
st.subheader('''PCA Settings:''')
pca_num_components_slider = st.slider('components', 2, 16, 2)
# perform pca
pca_reduction_result = cl_be.pca_dimensionality_reduction(gt_dataframe_preprocessed, pca_num_components_slider)

st.markdown('''***''')

# Clustering with PCA
st.header('''Clustering (after applying PCA):''')
st.subheader('''General Settings:''')
num_bins_slider = st.slider('Bins/ Classes within the data', 2, 16, 2)

st.markdown('''***''')

# KMeans Clustering
st.header('''KMEANS Clustering:''')
st.subheader('''KMEANS Settings:''')
random_state_slider_kmeans = st.slider('Select random state for KMEANS clustering:', 2, 16, 1)
# perform kmeans clustering
kmeans_colored_plot, correctly_colored_plot, kmeans_sil_co, true_kmeans_sil_co, kmeans_num_correct_labels, kmeans_correct_labels_percent = cl_be.perform_kmeans_clustering(pca_reduction_result, num_bins_slider, random_state_slider_kmeans, gt_preprocessed_labels_as_colors)
st.subheader('''KMEANS Results:''')
col3, col4, = st.columns([5, 5])
with col3:
    st.subheader('''KMEANS:''')
    kmeans_colored_plot
    st.write('silhouette coefficient: ', kmeans_sil_co)
    st.write('correct labels: ', kmeans_num_correct_labels, '/', number_datapoints)
    st.write('correct percent: ', kmeans_correct_labels_percent, '%')
with col4:
    st.subheader('''Ground Truth:''')
    correctly_colored_plot
    st.write('silhouette coefficient: ', true_kmeans_sil_co)

st.markdown('''***''')

# Spectral Clustering
st.header('''SPECTRAL Clustering:''')
st.subheader('''SPECTRAL Settings:''')
spectral_num_components_slider = st.slider('Number of Eigenvectors for SPECTRAL embedding:', 2, 16, 2)
random_state_slider_spectral = st.slider('Select random state for SPECTRAL clustering:', 2, 16, 2)
# perform spectral clustering
spectral_colored_plot, correctly_colored_plot, spectral_sil_co, true_spectral_sil_co, spectral_num_correct_labels, spectral_correct_labels_percent = cl_be.perform_spectral_clustering(pca_reduction_result, num_bins_slider, spectral_num_components_slider, random_state_slider_spectral, gt_preprocessed_labels_as_colors)
st.subheader('''SPECTRAL Results:''')
col5, col6, = st.columns([5, 5])
with col5:
    st.subheader('''SPECTRAL:''')
    spectral_colored_plot
    st.write('silhouette coefficient: ', spectral_sil_co)
    st.write('correct labels: ', spectral_num_correct_labels, '/', number_datapoints)
    st.write('correct percent: ', spectral_correct_labels_percent, '%')
with col6:
    st.subheader('''Ground Truth:''')
    correctly_colored_plot
    st.write('silhouette coefficient: ', true_spectral_sil_co)

st.markdown('''***''')

# DBSCAN Clustering
st.header('''DBSCAN Clustering:''')
st.subheader('''DBSCAN Settings:''')
DBSCAN_epsilon_slider = st.slider('Select the (epsilon * 10) value for DBSCAN clustering:', 1, 100, 10)
DBSCAN_min_samples_slider = st.slider('The number of samples (or total weight) in a neighborhood for a point to be considered as a core point:', 2, 20, 15)
# perform dbscan clustering
dbscan_colored_plot, correctly_colored_plot, dbscan_sil_co, true_dbscan_sil_co, dbscan_num_correct_labels, dbscan_correct_labels_percent = cl_be.perform_dbscan_clustering(pca_reduction_result, DBSCAN_epsilon_slider, DBSCAN_min_samples_slider, gt_preprocessed_labels_as_colors)
st.subheader('''DBSCAN Results:''')
col7, col8, = st.columns([5, 5])
with col7:
    st.subheader('''DBSCAN:''')
    dbscan_colored_plot
    st.write('silhouette coefficient: ', dbscan_sil_co)
    st.write('correct labels: ', dbscan_num_correct_labels, '/', number_datapoints)
    st.write('correct percent: ', dbscan_correct_labels_percent, '%')
with col8:
    st.subheader('''Ground Truth:''')
    correctly_colored_plot
    st.write('silhouette coefficient: ', true_dbscan_sil_co)

st.markdown('''***''')

# add ui elements to change settings for dimensionality reduction using tSNE
st.header('''Dimensionality reduction with tSNE:''')
st.subheader('''tSNE Settings:''')
tsne_num_components_slider = st.slider('tSNE components:', 2, 3, 2)
tsne_perplexity_slider = st.slider('tSNE perplexity:', 2, 100, 2)
tsne_iterations_slider = st.slider('tSNE iterations:', 100, 5000, 2000)
# perform tSNE
tsne_reduction_result = cl_be.tsne_dimensionality_reduction(gt_dataframe_preprocessed, tsne_num_components_slider, tsne_perplexity_slider, tsne_iterations_slider)

st.markdown('''***''')

# Clustering with PCA
st.header('''Clustering (after applying tSNE):''')
st.subheader('''General Settings:''')
tsne_num_bins_slider = st.slider('Bins/ Classes within the data:', 2, 16, 2)

st.markdown('''***''')

# tSNE Clustering
st.header('''KMEANS Clustering:''')
st.subheader('''KMEANS Settings:''')
tsne_random_state_slider_kmeans = st.slider('Select random state for KMEANS clustering:', 2, 16, 2)

# perform tSNE clustering
tsne_kmeans_colored_plot, tsne_correctly_colored_plot, tsne_kmeans_sil_co, tsne_true_kmeans_sil_co, tsne_kmeans_num_correct_labels, tsne_kmeans_correct_labels_percent = \
    cl_be.perform_kmeans_clustering(tsne_reduction_result, tsne_num_bins_slider, tsne_random_state_slider_kmeans, gt_preprocessed_labels_as_colors)
st.subheader('''KMEANS Results:''')
col3, col4, = st.columns([5, 5])
with col3:
    st.subheader('''KMEANS:''')
    tsne_kmeans_colored_plot
    st.write('silhouette coefficient: ', tsne_kmeans_sil_co)
    st.write('correct labels: ', tsne_kmeans_num_correct_labels, '/', number_datapoints)
    st.write('correct percent: ', tsne_kmeans_correct_labels_percent, '%')
with col4:
    st.subheader('''Ground Truth:''')
    tsne_correctly_colored_plot
    st.write('silhouette coefficient: ', tsne_true_kmeans_sil_co)

st.markdown('''***''')

# Spectral Clustering
st.header('''SPECTRAL Clustering:''')
st.subheader('''SPECTRAL Settings:''')
tsne_spectral_num_components_slider = st.slider('Number of Eigenvectors for SPECTRAL embedding', 2, 16, 2)
tsne_random_state_slider_spectral = st.slider('Select random state for SPECTRAL clustering', 2, 16, 2)
# perform spectral clustering
tsne_spectral_colored_plot, tsne_correctly_colored_plot, tsne_spectral_sil_co, tsne_true_spectral_sil_co, tsne_spectral_num_correct_labels, tsne_spectral_correct_labels_percent = cl_be.perform_spectral_clustering(tsne_reduction_result, tsne_num_bins_slider, tsne_spectral_num_components_slider, tsne_random_state_slider_spectral, gt_preprocessed_labels_as_colors)
st.subheader('''SPECTRAL Results:''')
col5, col6, = st.columns([5, 5])
with col5:
    st.subheader('''SPECTRAL:''')
    tsne_spectral_colored_plot
    st.write('silhouette coefficient: ', tsne_spectral_sil_co)
    st.write('correct labels: ', tsne_spectral_num_correct_labels, '/', number_datapoints)
    st.write('correct percent: ', tsne_spectral_correct_labels_percent, '%')
with col6:
    st.subheader('''Ground Truth:''')
    tsne_correctly_colored_plot
    st.write('silhouette coefficient: ', tsne_true_spectral_sil_co)

st.markdown('''***''')

# DBSCAN Clustering
st.header('''DBSCAN Clustering:''')
st.subheader('''DBSCAN Settings:''')
tsne_DBSCAN_epsilon_slider = st.slider('Select the (epsilon * 10) value for DBSCAN clustering', 1, 100, 12)
tsne_DBSCAN_min_samples_slider = st.slider('The number of samples (or total weight) in a neighborhood for a point to be considered as a core point', 2, 20, 18)
# perform dbscan clustering
tsne_dbscan_colored_plot, tsne_correctly_colored_plot, tsne_dbscan_sil_co, tsne_true_dbscan_sil_co, tsne_dbscan_num_correct_labels, tsne_dbscan_correct_labels_percent = cl_be.perform_dbscan_clustering(tsne_reduction_result, tsne_DBSCAN_epsilon_slider, tsne_DBSCAN_min_samples_slider, gt_preprocessed_labels_as_colors)
st.subheader('''DBSCAN Results:''')
col7, col8, = st.columns([5, 5])
with col7:
    st.subheader('''DBSCAN:''')
    tsne_dbscan_colored_plot
    st.write('silhouette coefficient: ', tsne_dbscan_sil_co)
    st.write('correct labels: ', tsne_dbscan_num_correct_labels, '/', number_datapoints)
    st.write('correct percent: ', tsne_dbscan_correct_labels_percent, '%')
with col8:
    st.subheader('''Ground Truth:''')
    tsne_correctly_colored_plot
    st.write('silhouette coefficient: ', tsne_true_dbscan_sil_co)
