# RAG Document Chat Application

## Overview
A Retrieval-Augmented Generation (RAG) application that allows users to chat with various document types using Azure OpenAI and Arize Phoenix for tracing.

## Features
- Support for multiple document types:
  - PDF
  - Text files
  - Excel sheets
  - CSV files
  - SQL files

- Azure OpenAI integration
- Vector store using FAISS
- Tracing and evaluation with Arize Phoenix

## Setup

### Prerequisites
- Python 3.9+
- Azure OpenAI account
- Arize Phoenix account (optional)

### Installation
1. Clone the repository
2. Create a virtual environment
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Configuration
1. Copy `.env.example` to `.env`
2. Fill in your Azure OpenAI credentials

### Running the Application
```
streamlit run ui/streamlit_app.py
```

## Usage
1. Configure Azure OpenAI settings
2. Upload a document
3. Start chatting with your document!