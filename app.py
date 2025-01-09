import streamlit as st
import requests


def main():
    st.title("Semantic News Search")

    query = st.text_input("Enter your search query:")
    print(query)
    if st.button("Search"):
        if query:
            response = requests.post(
                "http://localhost:5000/search", json={"query": query}
            )
            results = response.json()
            print(results)
            for idx, result in enumerate(results, 1):
                st.subheader(f"Result {idx}")
                st.write(f"Title: {result['metadata']['title']}")
                st.write(f"Content: {result['metadata']['content']}")
                st.write(f"Similarity Score: {result['distance']:.4f}")
                st.divider()


if __name__ == "__main__":
    main()
