import streamlit as st
from keycloak import KeycloakOpenID
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader
import google.generativeai as genai
import requests

# Keycloak Configuration
KEYCLOAK_SERVER_URL = "http://localhost:8181"
REALM_NAME = "myrealm"
CLIENT_ID = "streamlit-app"
REDIRECT_URI = "http://localhost:8501"

keycloak_openid = KeycloakOpenID(server_url=KEYCLOAK_SERVER_URL, client_id=CLIENT_ID, realm_name=REALM_NAME)

# Streamlit Authentication
st.title("Chat Application with CSV Files - Keycloak Authenticated")

if "access_token" not in st.session_state:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        try:
            token = keycloak_openid.token(username, password)
            st.session_state["access_token"] = token["access_token"]
            st.success("Login Successful!")
            st.experimental_rerun()
        except Exception as e:
            st.error("Invalid credentials. Try again.")

if "access_token" in st.session_state:
    st.sidebar.header("Use Cases")
    use_case = st.sidebar.selectbox(
        "Choose a Use Case",
        ["Summarization", "Question and Answer", "Text Generation"]
    )
    st.sidebar.write("Dataset: Healthcare Records")

    # Load Data & Model
    file_path = "healthcare_dataset.csv"
    loader = CSVLoader(file_path=file_path)
    data = loader.load()[:20]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents=data, embedding=embeddings)
    retriever = vector_store.as_retriever()

    api_key = "your-gemini-api-key"
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel("gemini-1.5-flash-latest")

    if use_case == "Summarization":
        st.header("Summarization")
        search_query = st.text_input("Enter a Search Query:")
        if st.button("Summarize"):
            if search_query.strip():
                retrieved_docs = retriever.get_relevant_documents(search_query)
                if retrieved_docs:
                    combined_content = " ".join([doc.page_content for doc in retrieved_docs])
                    summarization_prompt = f"Summarize the following Text:\n\n{combined_content}"
                    response = llm.generate_content(summarization_prompt)
                    st.subheader("Generated Summary")
                    st.write(response.text)

    elif use_case == "Question and Answer":
        st.header("Question and Answer")
        question_1 = st.text_input("Enter your Question:")
        if st.button("Get Answer"):
            if question_1.strip():
                retrieved_docs = retriever.get_relevant_documents(question_1)
                combined_content = " ".join([doc.page_content for doc in retrieved_docs])
                prompt_template = f"""
                Answer the question as detailed as possible from the provided context.
                If the answer is not in the provided context, say "answer is not available in the context".
                
                Context:
                {combined_content}
                
                Question:
                {question_1}
                
                Answer:
                """
                qa_result = llm.generate_content(prompt_template)
                st.subheader("Answer")
                st.write(qa_result.text)

    elif use_case == "Text Generation":
        st.header("Text Generation")
        prompt = st.text_input("Enter a prompt:")
        if st.button("Generate Text"):
            if prompt.strip():
                generated_text = llm.generate_content(prompt)
                st.subheader("Generated Text")
                st.write(generated_text.text)
            else:
                st.warning("Please enter a valid prompt!")
    
    if st.sidebar.button("Logout"):
        del st.session_state["access_token"]
        st.experimental_rerun()
