import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        else:
            return None, f"Unsupported file type: {file_extension}"

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        return texts, None
    except Exception as e:
        return None, str(e)

def create_vector_store(texts):
    try:
        # Use Azure OpenAI Embeddings
        embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT'),
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
            model=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
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
    st.title("📄 DocuChat Pro: Multi-Format Document Interaction Hub")

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

                with st.chat_message("assistant"):
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
