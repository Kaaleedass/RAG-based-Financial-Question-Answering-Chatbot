# Financial Knowledge Assistant — RAG Chatbot

An interactive financial question-answering chatbot built using Retrieval-Augmented Generation (RAG). The application retrieves relevant information from Zerodha Varsity educational articles and uses an OpenAI language model to generate concise answers.

## Project Overview

This project demonstrates how a RAG pipeline can connect a language model to an external knowledge source, improving its ability to answer questions using relevant retrieved information.

## Features

- Website content ingestion from a sitemap
- Document splitting using RecursiveCharacterTextSplitter
- Vector embeddings using OpenAI Embeddings
- Semantic retrieval with ChromaDB
- History-aware retrieval for follow-up questions
- Concise answers generated using GPT-4o mini
- Interactive chatbot interface built with Gradio

## Architecture

1. **Data ingestion:** Collect educational article URLs from the sitemap and load website content.
2. **Text splitting:** Divide documents into chunks of 1,000 characters with 200 characters of overlap.
3. **Embedding generation:** Convert text chunks into vector embeddings.
4. **Vector storage:** Store embeddings in ChromaDB.
5. **Question contextualization:** Use conversation history to reformulate follow-up questions.
6. **Retrieval and generation:** Retrieve relevant chunks and provide them to the language model.
7. **User interface:** Display answers through the Gradio chatbot.

## Technology Stack

- Python
- LangChain
- OpenAI GPT-4o mini
- OpenAI Embeddings
- ChromaDB
- BeautifulSoup
- Gradio

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/rag-financial-qa-chatbot.git
cd rag-financial-qa-chatbot
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Set your OpenAI API key as an environment variable in your terminal:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

Run the application:

```bash
python app.py
```

Open the local Gradio URL displayed in your terminal.

## Configuration

The current implementation loads the first 10 URLs from the configured sitemap. You can adjust the number of pages to process in the source code.

## Learning Outcomes

- Understanding Retrieval-Augmented Generation
- Building document ingestion and chunking pipelines
- Working with vector databases and embeddings
- Implementing history-aware retrieval
- Connecting LLMs to a conversational web interface

## Disclaimer

This project is intended for educational and informational purposes only. It does not provide personalized financial or investment advice. Always verify financial information using reliable sources.

## License

Choose an appropriate open-source license before distributing or reusing this project.
