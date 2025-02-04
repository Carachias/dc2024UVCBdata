"""Create a left-side navigation-bar that contains all the subpages of the app."""
# Library imports
import streamlit as st


# create the navigation bar
left_nav_bar = st.navigation([st.Page("welcome_page.py", title="Home"),
                              st.Page("dset_inspector.py", title="Inspect / Modify Data"),
                              st.Page("dset_clusterer.py", title="Clustering"),
                              st.Page("dset_classifier.py", title="Classify")])

# run and render the content of this entrypoint file
left_nav_bar.run()
