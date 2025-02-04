"""Start the data analysis app with streamlit."""
# Library imports
import subprocess


def main():
    """Main function to launch the data analysis app.
    This function creates a subprocess that runs the wrapper containing all the subpages of the app."""
    app_process = subprocess.Popen(["streamlit", "run", "pages_test/page_wrapper.py"], stdout=subprocess.PIPE)
    app_process.communicate()


if __name__ == "__main__":
    main()
