import streamlit as st
import json
import requests
import sys

if len(sys.argv) == 1:
    url = 'http://0.0.0.0:9696/predict'
else:
    url = f'{sys.argv[1]}/predict'

# Streamlit app
def main():
    st.title("Zoomcamp Response Helper")

    st.write("This is an App that Helps you find relevant information about the Machine Learning Zoomcamp and the Data Engineering Zoomcamp.")
    st.write("The questions and answers data were extracted from:") 
    st.markdown("- DTC Zoomcamp Q&A Challenge dataset")
    st.markdown("- Machine Learning Zoomcamp FAQ")
    st.markdown("- Data Engineering Zoomcamp FAQ")
    st.write("")
    st.write("You will Receive the Top 3 answers with the Highest Score according to our model")

    # Search bar (not functional in this example)
    query = st.text_input("Enter your search query:")

    # Button to trigger API call
    if st.button("Search"):

        query_json = [
            {
            "query" : f"{query}"
            }
        ]

        response = requests.post(url, json=query_json[0]).json()

        # Display each item in the JSON response
        for item in response['data']:
            st.text_area("Answer", value=item['answer'], height=150)
            st.write(f"Score: {item['score']}")
            st.write("---")  # Separator

if __name__ == "__main__":
    main()
