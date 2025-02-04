"""This file contains the frontend for the classifiers functionality of the app."""

# Library imports
import streamlit as st
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from imblearn.datasets import make_imbalance
from imblearn.over_sampling import RandomOverSampler, SMOTE, KMeansSMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids

# File imports
import backend.classifier_backend as be


# Draw a title and some text to the app:
st.title('''Data Classification''')
st.header('''Upload some Data:''')
st.subheader('''Ground-Truth:''')
st.markdown('''You can have a look at the Example below which is working with the two
largest classes in the Dataset only or modify the dataset with the dataset editor
page and upload the editors .csv output file here to use that:''')
upload_gt_file = st.file_uploader(label="Upload 'Ground Truth'-like data:", type='csv', accept_multiple_files=False)

if upload_gt_file is None:
    gt_dataframe = pd.read_csv("data/modified/two_2_largest_classes.csv")

if upload_gt_file is not None:
    gt_dataframe = pd.read_csv(upload_gt_file)

option_include_toxpi = st.selectbox("What kind of input data do you want to use?", ("GT only", "ToxPi only", "Combination"))
st.write("You selected:", option_include_toxpi)

# remove undesired features as well as useless datapoints (Zero-Rows) in case data modification tool was not used
if option_include_toxpi in ["ToxPi only", "Combination"]:
    toxpi_dataframe = pd.read_csv("data/qcpheno.out_results.csv")
    combined_dataframe = be.combine_dataframes(gt_dataframe, toxpi_dataframe)
    gt_dataframe_preprocessed, gt_preprocessed_labels, gt_preprocessed_labels_as_colors = be.perform_dataframe_preprocessing(combined_dataframe, option_include_toxpi)
if option_include_toxpi == "GT only":
    gt_dataframe_preprocessed, gt_preprocessed_labels, gt_preprocessed_labels_as_colors = be.perform_dataframe_preprocessing(gt_dataframe, option_include_toxpi)

st.header('''Preprocessed Data:''')
st.markdown('''- Transformed string Class labels into integers  
- Removed remaining string column Substance''')
gt_dataframe_preprocessed

st.header('''Creating training/ testing split:''')
st.markdown('''- 80% used for training  
    - will be modified (sub-/ oversampled) for each experiment
- 20% held back for testing
    - will be reused for all following experiments''')

# create training testing split and use the testing portion for all the experiments
# X_train, X_test, y_train, y_test = train_test_split(gt_dataframe_preprocessed, gt_preprocessed_labels_as_colors['Label'], test_size=0.2, shuffle=True)
# experimental self-made function to generate a training/ testing split to ensure smote works without errors
X_train, X_test, y_train, y_test = be.create_train_test_split(gt_dataframe_preprocessed, gt_preprocessed_labels_as_colors)


st.header('''Classifier Settings:''')
classifier_type_option = st.selectbox("what classifier type do you want to use?", ('SVM', 'MLP Classifier', 'Decision Tree'))
st.write("You selected:", classifier_type_option)
classifier_argument_list = []
if classifier_type_option == 'SVM':
    svm_kernel_option = st.selectbox("What kind of kernel you want to use?", ('linear', 'poly', 'rbf', 'sigmoid'))
    st.write("You selected:", svm_kernel_option)

    polydeg = 3
    if svm_kernel_option == 'poly':
        svm_poly_deg = st.slider("Set the desired degree for the polynomial kernel", 2, 10, 3)
        polydeg = svm_poly_deg
        st.write("You selected:", polydeg)

    gammaval = 'scale'
    if svm_kernel_option in ['poly', 'rbf', 'sigmoid']:
        svm_gamma_val = st.selectbox("What kind of gamma value do you want to use?", ('scale', 'auto'))
        gammaval = svm_gamma_val
        st.write("You selected:", gammaval)

    classifier_argument_list = [svm_kernel_option, polydeg, gammaval]

if classifier_type_option == 'MLP Classifier':
    activation_fun_option = st.selectbox("what activation function do you want to use?", ('relu', 'identity', 'logistic', 'tanh'))
    learning_rate_option = st.selectbox("what type of learning rate do you want to use?", ('constant', 'invscaling', 'adaptive'))
    learning_rate_init_slider = st.slider("Set the learning rate (slider value / 1000)", 1, 10, 1)
    max_iter_slider = st.slider("Set maximum amount of iterations", 20, 1000, 50)
    early_stopping_option = st.selectbox("Do you want to use early stopping?", ('False', 'True'))

    classifier_argument_list = [activation_fun_option, learning_rate_option, learning_rate_init_slider, max_iter_slider, early_stopping_option]

if classifier_type_option == 'Decision Tree':
    criterion_option = st.selectbox("What criterion do you want to use?", ('gini', 'entropy', 'log_loss'))
    """
    learning_rate_option = st.selectbox("what type of learning rate do you want to use?", ('constant', 'invscaling', 'adaptive'))
    learning_rate_init_slider = st.slider("Set the learning rate (slider value / 1000)", 1, 10, 1)
    max_iter_slider = st.slider("Set maximum amount of iterations", 20, 1000, 50)
    early_stopping_option = st.selectbox("Do you want to use early stopping?", ('False', 'True'))
    """
    classifier_argument_list = [criterion_option]

custom_classifier = be.create_and_return_classifier(classifier_type_option, classifier_argument_list)

st.markdown('''***''')

st.title('''Full Data (Baseline):''')
# train model on original ds
clf = custom_classifier
clf.fit(X_train, y_train)

# test model with regular testing ds
predictions = clf.predict(X_test)

# create test result dataframe
resultframe = be.create_test_result_frame(predictions, y_test)

# calculate overall quality measures
accuracy = accuracy_score(y_test, predictions, normalize=True, sample_weight=None)*100
f1_scoreval = f1_score(y_test, predictions, average='micro')

# calculate detailed quality measures:
report_dataframe, overall_results, class_based_results, distribution_info_frame = be.calculate_detailed_testing_result(resultframe)

# print everything to screen
st.header('''Training-Data (Detailed Info):''')
col5, col6, mid1, col7 = st.columns([16, 6, 1, 8])
with col5:
    st.write("Selected Datapoints: ", X_train)
with col6:
    st.markdown("Labels: ")
    st.dataframe(y_train, width=150)
with col7:
    st.write("Class Distribution:")
    with st.container(height=405, border=True):
        st.write("Original Data:", Counter(y_train), "\n\n")

st.header('''Testing:''')
col1, col2, = st.columns([10, 19])
with col1:
    with st.container(height=900, border=True):
        st.header('''Exact Model Predictions:''')
        st.dataframe(resultframe, width=200, height=595)
with col2:
    with st.container(height=900, border=True):
        st.header('''Overall Result:''')
        overall_results
        st.write("Classification Accuracy:", accuracy, "%")
        st.header('''Per Class Result:''')
        class_based_results
        st.header('''Class Distribution:''')
        st.dataframe(distribution_info_frame, column_config={"col0": st.column_config.NumberColumn(label="", width='small')})
        # st.dataframe(report_dataframe, column_config={"col0": st.column_config.NumberColumn(label="ID", width='large')})

st.markdown('''***''')

st.title('''Classification using Undersampling:''')
# create under-sampled ds
rand_usam = ClusterCentroids(random_state=15)
X_undersampled, Y_undersampled = rand_usam.fit_resample(X_train, y_train)

# train model on original ds
clf_us = custom_classifier
clf_us.fit(X_undersampled, Y_undersampled)

# test model with regular testing ds
predictions_us = clf_us.predict(X_test)

# create test result dataframe
resultframe_us = be.create_test_result_frame(predictions_us, y_test)

# calculate overall quality measures
accuracy_us = accuracy_score(y_test, predictions_us, normalize=True, sample_weight=None)*100
f1_scoreval_us = f1_score(y_test, predictions_us, average='micro')

# calculate detailed quality measures:
report_dataframe_us, overall_results_us, class_based_results_us, distribution_info_frame_us = be.calculate_detailed_testing_result(resultframe_us)

# print everything to screen
st.header('''Training-Data (Detailed Info):''')
col5, col6, mid1, col7 = st.columns([16, 6, 1, 8])
with col5:
    st.write("Selected Datapoints: ", X_undersampled)
with col6:
    st.markdown("Labels: ")
    st.dataframe(Y_undersampled, width=150)
with col7:
    st.write("Class Distribution:")
    with st.container(height=405, border=True):
        st.write("Undersampled Data:", Counter(Y_undersampled), "\n\n")

st.header('''Testing:''')
col1, col2, = st.columns([10, 19])
with col1:
    with st.container(height=900, border=True):
        st.header('''Exact Model Predictions:''')
        st.dataframe(resultframe_us, width=200, height=595)
with col2:
    with st.container(height=900, border=True):
        st.header('''Overall Result:''')
        overall_results_us
        st.write("Classification Accuracy:", accuracy_us, "%")
        st.header('''Per Class Result:''')
        class_based_results_us
        st.header('''Class Distribution:''')
        st.dataframe(distribution_info_frame_us, column_config={"col0": st.column_config.NumberColumn(label="", width='small')})
        # st.dataframe(report_dataframe, column_config={"col0": st.column_config.NumberColumn(label="ID", width='large')})

st.markdown('''***''')

st.title('''Classification using Oversampling:''')
st.header('''Select Oversampling method:''')
oversampling_method_option = st.selectbox("what oversampling method do you want to use?", ('random', 'SMOTE', 'KMeansSMOTE', 'BorderlineSMOTE', 'ADASYN'))
if oversampling_method_option == 'random':
    rand_osam = RandomOverSampler(random_state=10)
    X_oversampled, Y_oversampled = rand_osam.fit_resample(X_train, y_train)
if oversampling_method_option == 'SMOTE':
    smote_osam = SMOTE(random_state=10)
    X_oversampled, Y_oversampled = smote_osam.fit_resample(X_train, y_train)
if oversampling_method_option == 'KMeansSMOTE':
    KMeansSMOTE_osam = KMeansSMOTE(random_state=10, cluster_balance_threshold=0.15)
    X_oversampled, Y_oversampled = KMeansSMOTE_osam.fit_resample(X_train, y_train)
if oversampling_method_option == 'BorderlineSMOTE':
    BorderlineSMOTE_osam = BorderlineSMOTE(random_state=10)
    X_oversampled, Y_oversampled = BorderlineSMOTE_osam.fit_resample(X_train, y_train)
if oversampling_method_option == 'ADASYN':
    ADASYN_osam = ADASYN(random_state=10, sampling_strategy='not majority', n_neighbors=4)
    X_oversampled, Y_oversampled = ADASYN_osam.fit_resample(X_train, y_train)

# train model on original ds
clf_os = custom_classifier
clf_os.fit(X_oversampled, Y_oversampled)

# test model with regular testing ds
predictions_os = clf_os.predict(X_test)

# create test result dataframe
resultframe_os = be.create_test_result_frame(predictions_os, y_test)

# calculate overall quality measures
accuracy_os = accuracy_score(y_test, predictions_os, normalize=True, sample_weight=None)*100
f1_scoreval_os = f1_score(y_test, predictions_os, average='micro')

# calculate detailed quality measures:
report_dataframe_os, overall_results_os, class_based_results_os, distribution_info_frame_os = be.calculate_detailed_testing_result(resultframe_os)

# print everything to screen
st.header('''Training-Data (Detailed Info):''')
col5, col6, mid1, col7 = st.columns([16, 6, 1, 8])
with col5:
    st.write("Selected Datapoints: ", X_oversampled)
with col6:
    st.markdown("Labels: ")
    st.dataframe(Y_oversampled, width=150)
with col7:
    st.write("Class Distribution:")
    with st.container(height=405, border=True):
        st.write("Oversampled Data:", Counter(Y_oversampled), "\n\n")

st.header('''Testing:''')
col1, col2, = st.columns([10, 19])
with col1:
    with st.container(height=900, border=True):
        st.header('''Exact Model Predictions:''')
        st.dataframe(resultframe_os, width=200, height=595)
with col2:
    with st.container(height=900, border=True):
        st.header('''Overall Result:''')
        overall_results_os
        st.write("Classification Accuracy:", accuracy_os, "%")
        st.header('''Per Class Result:''')
        class_based_results_os
        st.header('''Class Distribution:''')
        st.dataframe(distribution_info_frame_os, column_config={"col0": st.column_config.NumberColumn(label="", width='small')})
        # st.dataframe(report_dataframe, column_config={"col0": st.column_config.NumberColumn(label="ID", width='large')})
