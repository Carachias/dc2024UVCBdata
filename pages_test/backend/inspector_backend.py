"""This file contains all the relevant and reusable functions for the data inspection/ modification tool."""

# Library imports
import pandas as pd
import matplotlib.pyplot as plt


def get_basic_dataset_properties(dataframe):
    """This function finds and returns very basic properties of a given dataframe.

    :param dataframe: input dataframe to get information about
    :return:
    """
    number_rows = dataframe.shape[0]
    num_bad_rows = len(dataframe[dataframe['mg'] == 0])
    num_good_rows = number_rows - num_bad_rows
    num_classes = (dataframe["Class"]).drop_duplicates().shape[0]

    return number_rows, num_bad_rows, num_good_rows, num_classes


def create_plot(dataframe, x_axis_label, y_axis_label):
    """Function to plot dataframes consisting of two columns.

    One column is meant to be used as values on the xAxis and the other
    is supposed to be used as the yAxis values.
    :param dataframe: input dataframe consisting of two columns
    :param x_axis_label: the column to be used for the xAxis
    :param y_axis_label: the column to be used as the yAxis
    :return: finished plot that can be saved or printed to screen
    """
    fig, ax = plt.subplots()
    ax.bar(dataframe[x_axis_label], dataframe[y_axis_label])
    ax.tick_params(axis='x', labelrotation=90)

    return fig


def create_mean_toxpi_df(dataframe):
    """Function to generate a dataframe of mean values concerning the ToxPi values.

    Dataframe will be passed to the plot function afterwards.
    Function returns finished plot.
    :param dataframe: input dataframe to generate mean values from
    :return:
    """
    # create correct dataframe
    mean_df = dataframe.groupby("Class", as_index=False)["ToxPi Score"].mean()

    # generate a plot
    mean_plot = create_plot(mean_df, "Class", "ToxPi Score")

    return mean_df, mean_plot


def create_median_toxpi_df(dataframe):
    """Function to generate a dataframe of median values concerning the ToxPi values.

    Dataframe will be passed to the plot function afterwards.
    Function returns finished plot.
    :param dataframe: input dataframe to generate median values from
    :return:
    """
    # create correct dataframe
    median_df = dataframe.groupby("Class", as_index=False)["ToxPi Score"].median()

    # generate a plot
    median_plot = create_plot(median_df, "Class", "ToxPi Score")

    return median_df, median_plot


def create_class_distribution_df(dataframe):
    """Function to generate a class distribution dataframe.

    Dataframe will be passed to the plot function afterwards.
    Function returns the finished plot.
    :param dataframe: input dataframe to create class distribution from
    :return:
    """
    # create correct dataframe
    distribution_lst = list(dataframe['Class'])
    value_counts = pd.Series(distribution_lst).value_counts(sort=False)
    distribution_df = pd.DataFrame(value_counts)
    distribution_df = distribution_df.reset_index()
    distribution_df.columns = ['Class', 'Amount']

    # generate a plot
    distribution_plot = create_plot(distribution_df, 'Class', 'Amount')

    return distribution_df, distribution_plot


def create_usability_row_distribution_df(dataframe, usability):
    """Function to generate a dataframe with the distribution of zero-rows.

    Dataframe will be passed to the plot function afterwards.
    Function returns the finished plot.
    :param dataframe: input dataframe to create zero-row distribution from
    :param usability: string to specify if filtering criteria is zero-rows or non-zero-rows
    :return:
    """
    # create correct dataframe
    distribution_df = pd.DataFrame(columns=['Class', 'mg'])
    distribution_df['Class'] = dataframe['Class']
    distribution_df['mg'] = dataframe['mg']
    if usability == "useless":
        distribution_df.drop(distribution_df.loc[distribution_df['mg'] != 0].index, inplace=True)
    if usability == "useful":
        distribution_df.drop(distribution_df.loc[distribution_df['mg'] == 0].index, inplace=True)

    zero_row_distribution_df = distribution_df.groupby('Class', as_index=False)['mg'].count()
    zero_row_distribution_df.columns = ['Class', 'Amount']

    # generate a plot
    zero_row_distribution_plot = create_plot(zero_row_distribution_df, 'Class', 'Amount')

    return zero_row_distribution_df, zero_row_distribution_plot


def identify_small_classes(dataframe, threshold):
    """This function finds classes with less datapoints than a specified threshold.

    The labels of the classes with a number of rows smaller than the given threshold
    are added to a list. This List of classes/ class-labels is then returned.
    :param dataframe: input dataframe
    :param threshold: threshold for minimum class size
    :return:
    """
    # create a list of unique class labels
    class_distribution_list = list(dataframe["Class"])
    unique_class_labels_list = list((dataframe["Class"]).drop_duplicates())

    # create a list of classes (labels) smaller than the specified threshold
    drop_class_list = []

    for class_label in unique_class_labels_list:
        if class_distribution_list.count(class_label) < threshold:
            drop_class_list.append(class_label)

    return drop_class_list


def drop_classes_in_list(dataframe, drop_list):
    """This function removes classes specified in the drop_list from the dataframe.

    The resulting dataframe without the classes from the drop_list is then returned.
    :param dataframe: input dataframe before deletion of classes
    :param drop_list: list of class_labels that are going to be removed
    :return:
    """
    # modify input dataframe and remove all rows with labels from the drop_list
    modified_dataframe = dataframe
    for class_label in drop_list:
        modified_dataframe.drop(modified_dataframe.loc[modified_dataframe['Class'] == class_label].index, inplace=True)

    return modified_dataframe


def csvify_df(modified_dataframe):
    """Function to save the modified dset.

    :param modified_dataframe: input dataframe before deletion of classes
    :return:
    """
    return modified_dataframe.to_csv(index=False).encode('utf-8')
