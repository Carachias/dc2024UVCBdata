"""This file contains a home- or landing-page for the data analysis app."""
# Library imports
import streamlit as st


# render Text with information about the project itself
st.title('''Welcome!''')
st.header('''What this is:''')
st.text('''This is a small data analysis app, created as part of the 'Data Challenges' course at the Goethe University in Frankfurt during the winter semester 2024/2025.''')
st.markdown('''The purpose of the app is to provide an easy way to inspect, modify and analyze the UVCB data discussed in the following paper called:  
"Grouping  of UVCB Substances with New Approach Methodologies (NAMs) Data"''')

st.markdown('''***''')

# render Text with information about the apps capabilities
st.header('''App Features:''')
st.markdown('''The app consists of multiple smaller subpages that aim to provide the following functionality:''')
st.subheader('''Manipulating the data:''')
st.markdown('''- inspect the data and visualize important properties
- modify single datapoints
- select subsets of the data
- remove erroneous datapoints
- save the modified data to a new .csv file''')
st.subheader('''Clustering the data:''')
st.markdown('''Performing dimensionality reduction using:
- PCA (Principal Component Analysis)
- tSNE (t-Stochastic Neighborhood Embedding)''')
st.markdown('''Performing clustering using:
- kmeans
- spectral clustering
- dbscan''')
st.subheader('''Classifying the data:''')
st.markdown('''Training and testing Classifiers using your custom dataset:
- Support Vector Machine:  
    - linear kernel  
    - polynomial kernel with varible degree
    - rbf kernel
    - sigmoid kernel
- MLP Classifier:
    - Adjustable learning Rate
    - Early Stopping Parameter
    - Different Activation Functions
- Decision Tree
    - Selectable Criterion''')

st.markdown('''Different Oversampling Methods:
- Random Oversampling
- SMOTE
- KMeansSMOTE
- BorderlineSMOTE
- ADASYN''')

st.markdown('''Detailed Testing results:
- Summarized precision, recall and f1-score
- Summarized Classification accuracy
- Per Class precision, recall and f1-score''')

st.markdown('''***''')

st.header('''Knows Issues and Bugs:''')
st.subheader('''Clustering using DBSCAN:''')
st.markdown('''some parameter combinations of epsilon and n_uneighbors result in no clusters except noise or 
wrong ammount of clusters.
- this can lead to exception "invalid entry in choice array"''')
st.subheader('''Oversampling using ADASYN:''')
st.markdown('''Caused by the random selection of the training / testing split ADASYN might throw an error:
- "No samples will be generated with the provided ratio settings."''')


st.markdown('''***''')

# render Text with information about the used Sources
st.header('''Sources:''')
st.markdown('''Original Paper:  
https://www.altex.org/index.php/altex/article/view/1994/2187''')
st.markdown('''The original data can be obtained through the following link:    
https://github.com/ToxPi/ToxPi-Weight-Optimization/tree/main/uvcb_analysis''')
st.text('''''')
st.text('''''')

st.markdown('''Or more precisely:''')
st.markdown('''For the "Ground Truth" Data, visit:  
https://github.com/ToxPi/ToxPi-Weight-Optimization/blob/main/uvcb_analysis/1994-House_SupTab4.csv''')
st.markdown('''And for the ToxPi analysis performed in the mentioned paper:  
https://github.com/ToxPi/ToxPi-Weight-Optimization/blob/main/uvcb_analysis/qcpheno.out_results.csv''')
