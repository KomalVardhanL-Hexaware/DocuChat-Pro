import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import sqlparse
from langchain.schema import Document
import pandas as pd

#This is a dummy commit 

# Load environment variables
load_dotenv()

# Custom SQL Loader
class SQLLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        with open(self.file_path, 'r') as file:
            content = file.read()
        # Parse the SQL content
        parsed = sqlparse.parse(content)
        # Convert parsed SQL statements to a list of documents
        documents = [Document(page_content=str(statement)) for statement in parsed]
        return documents

# Custom CSV Loader using pandas
class CSVLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        for encoding in encodings:
            try:
                df = pd.read_csv(self.file_path, encoding=encoding)
                # Convert DataFrame to a list of documents
                documents = [Document(page_content=df.to_string(index=False))]
                return documents
            except Exception as e:
                st.sidebar.warning(f"Error loading CSV file with encoding {encoding}: {str(e)}")
        raise Exception(f"Error loading CSV file: Unable to decode with any of the tried encodings.")

# Utility functions
def load_and_split_document(file_path, file_extension):
    try:
        # Load document based on file type
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
        elif file_extension == '.csv':
            loader = CSVLoader(file_path)
        elif file_extension == '.sql':
            loader = SQLLoader(file_path)
        else:
            return None, f"Unsupported file type: {file_extension}"

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        return texts, None
    except Exception as e:
        return None, f"Error loading {file_path}: {str(e)}"

def create_vector_store(texts):
    try:
        # Use Azure OpenAI Embeddings
        embeddings = AzureOpenAIEmbeddings(
            deployment="text-embedding-3-large",
        )
        vectorstore = FAISS.from_documents(texts, embeddings)
        return vectorstore, None
    except Exception as e:
        return None, str(e)

def create_qa_chain(vectorstore):
    try:
        from langchain.chat_models import AzureChatOpenAI
        # Use Azure-specific configuration
        llm = AzureChatOpenAI(
            model="gpt-4o",
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        return qa_chain, None
    except Exception as e:
        return None, str(e)

# Streamlit UI
def main():
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            background-color: #f0f8ff;
        }
        .sidebar .sidebar-content {
            background-color: #e6f7ff;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            height: 3em;
            width: 100%;
            border-radius: 5px;
            border: 2px solid #008CBA;
            font-size: 20px;
        }
        .stTextInput>div>div>input {
            background-color: black;
            border: 2px solid #008CBA;
            border-radius: 5px;
            font-size: 20px;
        }
        .stChatMessage {
            background-color: black;
            border: 2px solid #008CBA;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        .header {
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            text-align: center;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="header"><h1>📄 DocuChat Pro: Multi-Format Document Interaction Hub</h1></div>', unsafe_allow_html=True)

    # Sidebar for document management
    st.sidebar.title("Upload Documents")
    uploaded_files = st.sidebar.file_uploader("Upload documents", type=['pdf', 'txt', 'csv', 'sql'], accept_multiple_files=True)

    if uploaded_files:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)

        all_texts = []
        for uploaded_file in uploaded_files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            file_path = os.path.join(temp_dir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            texts, error = load_and_split_document(file_path, file_extension)

            if error:
                st.sidebar.error(error)
                continue

            all_texts.extend(texts)

        if all_texts:
            with st.spinner("Creating vector store..."):
                vectorstore, error = create_vector_store(all_texts)

            if error:
                st.error(error)
                return

            with st.spinner("Initializing QA chain..."):
                qa_chain, error = create_qa_chain(vectorstore)

            if error:
                st.error(error)
                return

            st.success("Documents processed successfully!")

            # Chat interface
            if 'messages' not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            user_question = st.chat_input("Ask a question about your documents:")

            if user_question:
                with st.chat_message("user"):
                    st.markdown(user_question)
                    st.session_state.messages.append({"role": "user", "content": user_question})

                with st.chat_message("DocuChat Pro"):
                    with st.spinner("Generating response..."):
                        try:
                            response = qa_chain.invoke(user_question)
                            formatted_response = format_response_as_markdown(response['result'])
                            st.markdown(formatted_response)
                            st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                        except Exception as e:
                            st.error(f"Error generating response: {e}")

def format_response_as_markdown(response):
    # Dynamically format the response as Markdown
    formatted_response = f"""
    {response}
    """
    return formatted_response

if __name__ == "__main__":
    main()
