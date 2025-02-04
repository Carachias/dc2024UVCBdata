"""This file contains all the relevant and reusable functions for the data classification tool."""

# Library imports
import pandas as pd
import numpy as np
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


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

    no_zero_rows_class_labels_as_colors = pd.DataFrame(columns=['Label'])
    no_zero_rows_class_labels_as_colors['Label'] = label_encoder.transform(no_zero_rows_class_labelframe.Class)

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


def create_train_test_split(X_dataframe, Y_dataframe):
    """Generate a Training Testing Split where the smallest class in the testing potion ha at least the same
     amount of examples as there are classes.

    For example if we use 6 classes redraw the split until the smallest class in the training set contains
    at least 6 examples. This is necessary for Oversampling Methods like SMOTE to work. And due to the given Dataset
    being incredibly small this requirement often is not met on the first try.
    :param X_dataframe:
    :param Y_dataframe:
    :return:
    """
    # number of classes in the used data
    num_classes = (Y_dataframe['Label']).drop_duplicates().shape[0]

    # redraw the training/ testing split until the described size requirements are met
    smallest_class_in_train_split = 0
    while smallest_class_in_train_split < num_classes:
        X_train, X_test, y_train, y_test = train_test_split(X_dataframe, Y_dataframe['Label'], test_size=0.2, shuffle=True)
        smallest_class_in_train_split = min(Counter(y_train).values())

    return X_train, X_test, y_train, y_test


def create_test_result_frame(model_predictions, true_labels):
    """Create a dataframe containing a models predictions vs the actual labels and return it.

    :param model_predictions:
    :param true_labels:
    :return:
    """
    # create test result dataframe
    resultframe = pd.DataFrame(columns=["Labels", "predict"])
    resultframe['Labels'] = true_labels
    resultframe['predict'] = model_predictions

    return resultframe


def calculate_detailed_testing_result(resultframe):
    """Create multiple detailed testing results from a dataframe containing labels and predictions.

    :param resultframe: input frame containing the labels and predictions
    :return:
    """
    # generate target names:
    num_target_names = len(list(dict.fromkeys((list(resultframe["Labels"].drop_duplicates()) + list(resultframe["predict"].drop_duplicates())))))
    target_names = []
    for i in range(0, num_target_names):
        target_names.append('Class ' + str(i))

    # create detailed classification report and convert it to a readable dataframe
    classification_result_report = classification_report(resultframe["Labels"], resultframe["predict"], target_names=target_names, zero_division=0.0, output_dict=True)
    report_dataframe = pd.DataFrame.from_dict(classification_result_report, orient='columns').drop(['accuracy'], axis=1).T

    # split full frame into overall and class based frames
    class_based_results = report_dataframe.iloc[:len(target_names), :]
    overall_results = report_dataframe.iloc[len(target_names):, :]
    class_distribution_frame = class_based_results.drop(['precision', 'recall', 'f1-score'], axis=1).rename(columns={"support": "amount"}).T

    return report_dataframe, overall_results, class_based_results, class_distribution_frame


def create_and_return_classifier(classifier_type, argument_list):
    """Create and return a classifier with the given attributes.

    :param classifier_type: specifying if classifier should be of type SVM or MLP
    :param argument_list: list of arguments and settings for the given classifier type
    :return:
    """
    if classifier_type == 'SVM':
        custom_classifier = SVC(kernel=argument_list[0], degree=argument_list[1], gamma=argument_list[2])
    if classifier_type == 'MLP Classifier':
        custom_classifier = MLPClassifier(activation=argument_list[0], learning_rate=argument_list[1], learning_rate_init=(argument_list[2] / 1000),
                                          max_iter=argument_list[3], early_stopping=bool(argument_list[4]))
    if classifier_type == 'Decision Tree':
        custom_classifier = DecisionTreeClassifier(criterion=argument_list[0])

    return custom_classifier
