# HCA RAG Knowledge Assistant

## Project Overview

The HCA RAG Knowledge Assistant is a Retrieval-Augmented Generation (RAG) based AI chatbot developed for Hari Chand Anand & Co. The system answers user queries using company-specific documents instead of relying on general AI knowledge.

The chatbot retrieves relevant information from the company's knowledge base and uses a Large Language Model (LLM) to generate accurate, context-aware responses.

---

## Objective

The objective of this project is to build an intelligent knowledge assistant that can:

* Answer questions based on company documents
* Retrieve relevant information using semantic search
* Improve accuracy through document-based responses
* Demonstrate the implementation of a complete RAG pipeline

---

## Technologies Used

### Programming Language

* Python

### Frameworks & Libraries

* LangChain
* Streamlit
* ChromaDB
* HuggingFace Embeddings
* Google Gemini API
* Python Dotenv

### Embedding Model

* BAAI/bge-small-en-v1.5

### Vector Database

* ChromaDB

### LLM

* Gemini 2.0 Flash

---

## Project Structure

```text
HCA_RAG_Project
│
├── data
│   └── hca_company_knowledge_base.txt
│
├── vector_store
│
├── ingest.py
├── rag_chatbot.py
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

## Project Workflow

### Step 1: Document Loading

The company knowledge base is loaded using LangChain's TextLoader.

### Step 2: Text Chunking

The document is divided into smaller chunks using RecursiveCharacterTextSplitter.

### Step 3: Embedding Generation

Each chunk is converted into vector embeddings using the HuggingFace embedding model.

### Step 4: Vector Database Creation

Embeddings are stored in ChromaDB for efficient semantic search.

### Step 5: Retrieval

When a user asks a question, the most relevant chunks are retrieved from ChromaDB.

### Step 6: Answer Generation

The retrieved context is passed to Gemini, which generates an answer based only on the company knowledge base.

---

## System Architecture

```text
Company Knowledge Base
          │
          ▼
     Text Chunking
          │
          ▼
   Embedding Generation
          │
          ▼
       ChromaDB
          │
          ▼
     User Question
          │
          ▼
   Semantic Retrieval
          │
          ▼
      Gemini LLM
          │
          ▼
      Final Answer
```

---

## Features

* Company-specific question answering
* Semantic search using vector embeddings
* Retrieval-Augmented Generation (RAG)
* Streamlit-based user interface
* Context-aware responses
* Fast document retrieval using ChromaDB

---

## Sample Questions

* Who founded Hari Chand Anand & Co.?
* What is the company's vision?
* What industries does HCA serve?
* What products and services does HCA provide?
* What after-sales support does HCA offer?

---

## Installation

### Clone Repository

```bash
git clone <repository_url>
cd HCA_RAG_Project
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

## Running the Project

### Generate Embeddings

```bash
python ingest.py
```

```bash
python vector_store.py
```
### Run Chatbot

```bash
python rag_chatbot.py
```

### Run Streamlit Application

```bash
streamlit run app.py
```

---

## Current Status

✅ Knowledge Base Created

✅ Document Chunking Completed

✅ Embeddings Generated

✅ ChromaDB Integrated

✅ Semantic Retrieval Working

✅ Gemini Integration Completed

✅ Streamlit UI Developed

✅ End-to-End Testing Completed

---

## Future Enhancements

* Support for multiple PDF documents
* Cloud deployment
* Chat history feature
* Voice-based interaction
* Document upload functionality
* Advanced search and filtering

---

## Developed By

**Shreyosi Konar**

AI Intern

Hari Chand Anand & Co.

2026
