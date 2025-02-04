"""This file contains the functionality to upload, inspect and manipulate a dataset and save the result to a new .csv file keeping the structure of the original dataset."""

# Library imports
import streamlit as st
import pandas as pd

# File imports
import backend.inspector_backend as be


# Draw a title and some text to the app:
st.title('''Data Inspection and Manipulation''')
st.header('''Upload some Data:''')
st.subheader('''Ground-Truth:''')
st.markdown('''To use this part of the app, you have to upload any dataset that comes in the shape
of the "Ground Truth"-data, meaning you can either upload the full dataset provided by the paper
or a custom subset that was created using this site:''')
upload_gt_file = st.file_uploader(label="Upload 'Ground Truth'-like data:", type='csv', accept_multiple_files=False)

st.subheader('''ToxPi:''')
st.markdown('''Additional functionality becomes available once you also upload a file containing
data in the shape of the ToxPi table.''')
upload_toxpi_file = st.file_uploader(label="Upload ToxPi-like data:", type='csv', accept_multiple_files=False)
st.markdown('''***''')

st.header('''Dataset Properties:''')
if upload_gt_file is None:
    # placeholder information as long as no gt file has been uploaded
    st.markdown('''Once you upload a data file, properties of the uploaded data will appear right here.''')

if upload_gt_file is not None:
    gt_dataframe = pd.read_csv(upload_gt_file)
    class_distribution_dataframe, class_distribution_plot = be.create_class_distribution_df(gt_dataframe)
    number_rows, num_bad_rows, num_good_rows, num_classes = be.get_basic_dataset_properties(gt_dataframe)
    zero_row_distribution_dataframe, zero_row_distribution_plot = be.create_usability_row_distribution_df(gt_dataframe, "useless")
    non_zero_row_distribution_dataframe, non_zero_row_distribution_plot = be.create_usability_row_distribution_df(gt_dataframe, "useful")

    # create session-state entry
    if "gt_dataframe" not in st.session_state:
        st.session_state.gt_dataframe = gt_dataframe

    # plot info to screen
    st.subheader('''General Info:''')
    st.write('Rows/ datapoints in the Dataset:', number_rows)
    st.write('Useless zero-rows in the Dataset:', num_bad_rows)
    st.write('Usable non-zero-rows in the Dataset:', num_good_rows)
    st.write('Number of classes in the dataset:', num_classes)

    st.subheader('''Class distribution:''')
    col1, col2 = st.columns([5, 2])
    with col1:
        st.markdown('''Data as plot:''')
        class_distribution_plot
    with col2:
        st.markdown('''Data as frame:''')
        class_distribution_dataframe

    st.subheader('''Zero-Rows per Class:''')
    col3, col4 = st.columns([5, 2])
    with col3:
        st.markdown('''Data as plot:''')
        zero_row_distribution_plot
    with col4:
        st.markdown('''Data as frame:''')
        zero_row_distribution_dataframe

    st.subheader('''Usable Rows per Class:''')
    col5, col6 = st.columns([5, 2])
    with col5:
        st.markdown('''Data as plot:''')
        non_zero_row_distribution_plot
    with col6:
        st.markdown('''Data as frame:''')
        non_zero_row_distribution_dataframe

if upload_toxpi_file is None:
    # placeholder information as long as no toxpi file has been uploaded
    st.markdown('''***''')
    st.markdown('''Once you upload a ToxPi file, more properties will be displayed.''')

if upload_toxpi_file is not None:
    toxpi_dataframe = pd.read_csv(upload_toxpi_file)

    # joining the two relevant dataframes
    joined_df = gt_dataframe.merge(toxpi_dataframe, how='inner', left_on="Substance", right_on="Name")
    reduced_toxpi_df = joined_df[["Class", "ToxPi Score"]]

    # perform calculations on the joined dataframe
    mean_toxpi_dataframe, mean_toxpi_plot = be.create_mean_toxpi_df(reduced_toxpi_df)
    median_toxpi_dataframe, median_toxpi_plot = be.create_median_toxpi_df(reduced_toxpi_df)

    #create session state entry
    if "toxpi_dataframe" not in st.session_state:
        st.session_state.toxpi_dataframe = toxpi_dataframe

    #plot info to screen
    st.subheader('''Mean ToxPi per Class:''')
    col1, col2 = st.columns([5, 2])
    with col1:
        st.markdown('''Data as plot:''')
        mean_toxpi_plot
    with col2:
        st.markdown('''Data as frame:''')
        mean_toxpi_dataframe

    st.subheader('''Median ToxPi per Class:''')
    col3, col4 = st.columns([5, 2])
    with col3:
        st.markdown('''Data as plot:''')
        median_toxpi_plot
    with col4:
        st.markdown('''Data as frame:''')
        median_toxpi_dataframe

st.markdown('''***''')
st.header('''Dataset Editor:''')
st.markdown('''This Editor is meant to be used with the Ground-Truth shaped data.''')
if upload_gt_file is None:
    # placeholder information as long as no gt file has been uploaded
    st.markdown('''Please upload a File to use the Dataset editor.''')

if upload_gt_file is not None:
    # initialize a session state variable that saves the modified data.
    if "modified_dataframe" not in st.session_state:
        st.session_state.modified_dataframe = gt_dataframe

    col7, col8, = st.columns([3, 1])
    with col7:
        with st.container(height=650, border=True):
            # show preview of the full ground_truth table before editing it
            st.markdown('''Full Data pre Class deletion:''')
            st.dataframe(gt_dataframe, height=400)
            # add threshold-slider to remove classes below the specified threshold
            st.markdown('''Use this slider to remove all classes smaller than this threshold:''')
            threshold_slider = st.slider('Threshold:', 0, 5, 40, key="thresholdslider")
    with col8:
        with st.container(height=650, border=True):
            # show preview of classes to be removed
            st.markdown('''Classes below threshold:''')
            drop_classes_list = be.identify_small_classes(gt_dataframe, threshold_slider)
            drop_classes_list

    with st.container(height=700, border=True):
        # show the table after editing it
        st.markdown('''Remaining Data after Class deletion:''')
        modified_dataframe = be.drop_classes_in_list(gt_dataframe, drop_classes_list)
        remaining_data_editor = st.data_editor(modified_dataframe, num_rows="dynamic", key="editable_data")

        # remove empty rows before downloading
        remaining_data_editor.drop(remaining_data_editor.loc[remaining_data_editor['mg'] == 0].index, inplace=True)

        # save the remaining data
        st.markdown('''Download the modified data as .csv file. Zero rows are removed before saving.''')
        final_dataset_as_csv = be.csvify_df(remaining_data_editor)
        final_data_download_filename = st.text_input("Enter a filename (press enter to confirm) and click the download button")
        st.download_button("Press to Download", final_dataset_as_csv, (final_data_download_filename + ".csv"), "data/modified", key='download-csv')
