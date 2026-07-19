# AI Customer Support RAG Chatbot

## Overview

This project is an AI-powered customer support chatbot built to answer frequently asked questions using Retrieval-Augmented Generation (RAG). Instead of relying only on a language model's existing knowledge, the chatbot retrieves relevant information from company documentation and website content to provide more accurate, up-to-date responses.

## Features

- Answers customer questions using company knowledge
- Retrieves relevant information from a vector database
- Uses semantic search for context-aware responses
- Reduces repetitive customer support inquiries
- Easily updated by re-indexing new website content

## Tech Stack

- Python
- Google Gemini
- LangChain
- ChromaDB
- Hugging Face Embeddings
- BeautifulSoup (web scraping)

## How It Works

1. Scrape and collect website content.
2. Split the text into smaller chunks.
3. Generate embeddings for each chunk.
4. Store embeddings in ChromaDB.
5. When a user asks a question:
   - Retrieve the most relevant documents.
   - Pass the retrieved context and question to Gemini.
   - Return a response grounded in company information.

## Project Structure

```
.
├── build_index.py      # Creates the vector database
├── chat.py             # Chat interface
├── scraper.py          # Scrapes website content
├── chroma_db/          # Vector database
├── data/               # Scraped documents
└── README.md
```

