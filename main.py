import os
import streamlit as st
import pickle
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

file_path = "faiss_store_groq.pkl"
main_placeholder = st.empty()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=500
)

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url.strip():
        urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs")

if process_url_clicked:
    if not urls:
        st.error("Please enter at least one valid URL.")
        st.stop()

    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data loading started...")
    data = loader.load()

    st.write("Loaded documents:", len(data))

    if not data:
        st.error("No data loaded. Try another article URL.")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ",", " ", ""],
        chunk_size=1500,
        chunk_overlap=300
    )

    main_placeholder.text("Text splitting started...")
    docs = text_splitter.split_documents(data)

    st.write("Created chunks:", len(docs))

    if not docs:
        st.error("No chunks created. URL text could not be extracted.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    main_placeholder.text("Embedding vector building started...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    time.sleep(1)
    main_placeholder.success("FAISS index saved successfully.")

query = st.text_input("Question:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
        )

        result = chain.invoke({"question": query})

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")

        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    else:
        st.error("Please process URLs first.")