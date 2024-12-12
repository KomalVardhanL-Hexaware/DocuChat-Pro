import os
import streamlit as st
import phoenix as px
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI


# Load environment variables
load_dotenv()


class DocumentChatAssistant:
    def __init__(self):
        # Initialize Phoenix tracing
        # px.init_tracing()
        # LangChainInstrumentor().instrument()

        # Azure OpenAI Configuration
        self.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.azure_key = os.getenv('AZURE_OPENAI_KEY')
        self.azure_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')

    def load_document(self, uploaded_file):
        """Load and split document"""
        # Create temp directory if not exists
        os.makedirs("temp", exist_ok=True)

        # Save uploaded file
        temp_file_path = os.path.join("temp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Determine loader based on file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(temp_file_path)
            elif file_extension in ['.txt', '.csv']:
                loader = TextLoader(temp_file_path)
            else:
                st.error(f"Unsupported file type: {file_extension}")
                return None

            # Load and split documents
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)

            return texts
        except Exception as e:
            st.error(f"Error loading document: {e}")
            return None

    def create_vector_store(self, texts):
        """Create Chroma vector store with Azure OpenAI embeddings"""
        embeddings = OpenAIEmbeddings(
            openai_api_type="azure",
            openai_api_base=self.azure_endpoint,
            openai_api_key=self.azure_key,
            deployment=self.azure_deployment
        )

        # Create in-memory Chroma vector store
        vectorstore = Chroma.from_documents(texts, embeddings)
        return vectorstore

    def create_qa_chain(self, vectorstore):
        """Create QA chain with Azure OpenAI"""
        llm = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            openai_api_key=self.azure_key,
            azure_api_version="2024-02-01"
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        return qa_chain


def main():
    st.title("📄 Azure Document Chat Assistant")

    # Initialize assistant
    assistant = DocumentChatAssistant()

    # Sidebar for Azure OpenAI Configuration
    st.sidebar.header("Azure OpenAI Configuration")
    azure_endpoint = st.sidebar.text_input(
        "Azure OpenAI Endpoint",
        value=os.getenv('AZURE_OPENAI_ENDPOINT', '')
    )
    azure_key = st.sidebar.text_input(
        "Azure OpenAI Key",
        type="password",
        value=os.getenv('AZURE_OPENAI_KEY', '')
    )
    azure_deployment = st.sidebar.text_input(
        "Azure OpenAI Deployment",
        value=os.getenv('AZURE_OPENAI_DEPLOYMENT', '')
    )

    # Phoenix Tracing Link
    phoenix_session = px.launch_app()
    st.sidebar.markdown(f"[Open Phoenix Tracing]({phoenix_session.url})")

    # Document Upload
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=['pdf', 'txt', 'csv']
    )

    # Vectorstore and QA Chain placeholders
    vectorstore = None
    qa_chain = None

    if uploaded_file:
        # Set Azure OpenAI environment variables
        os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
        os.environ["AZURE_OPENAI_KEY"] = azure_key
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = azure_deployment

        # Process Document
        with st.spinner("Processing document..."):
            texts = assistant.load_document(uploaded_file)

        if texts:
            # Create Vector Store
            with st.spinner("Creating vector store..."):
                vectorstore = assistant.create_vector_store(texts)

            # Create QA Chain
            qa_chain = assistant.create_qa_chain(vectorstore)

            st.success("Document processed successfully!")

    # Chat Interface
    if qa_chain:
        user_question = st.text_input("Ask a question about your document:")

        if user_question:
            with st.spinner("Generating response..."):
                try:
                    response = qa_chain.run(user_question)
                    st.write("Response:", response)
                except Exception as e:
                    st.error(f"Error generating response: {e}")


if __name__ == "__main__":
    main()