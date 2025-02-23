# Medical ChatBot

This project implements an AI-powered Medical Chatbot that leverages the Gemini 1.5 Flash model for generating responses based on medical PDF content.  It utilizes Retrieval-Augmented Generation (RAG) to provide accurate and contextually relevant information.  Pinecone Vector DB is used for efficient semantic search and retrieval, and the chatbot is deployed using Flask.  LangChain facilitates seamless LLM orchestration and real-time query handling.

## Project Overview

This Medical Chatbot allows users to ask questions related to the content within a provided medical PDF.  The system uses the following components:

* **Gemini 1.5 Flash:** The core large language model responsible for generating responses.  This project uses the Gemini API for inference.
* **LangChain:**  A framework for developing applications powered by language models. It simplifies the integration and management of different components, including the LLM and vector database.
* **Pinecone:** A vector database that stores embeddings of the PDF content, enabling efficient semantic search for relevant information.
* **Flask:** A web framework for creating the chatbot interface and handling user interactions.
* **Retrieval-Augmented Generation (RAG):** A technique that combines the power of LLMs with external knowledge sources.  In this project, RAG retrieves relevant chunks of text from the medical PDF based on the user's query before passing it to the LLM.

## Features

* **Accurate Responses:** Leverages RAG and semantic search to provide contextually relevant answers derived from the medical PDF.
* **Efficient Retrieval:** Pinecone Vector DB enables fast and accurate retrieval of relevant information.
* **Real-time Interaction:** Flask powers a dynamic web interface for real-time communication with the chactbot.
* **User-Friendly Interface:**  A simple and intuitive chat interface for easy interaction.

## Installation

1. **Clone the repository:**

```bash
git clone [https://github.com/](https://github.com/)[your_github_username]/medical-chatbot.git  # Replace with your repo URL
cd medical-chatbot
```

2. **Setting up the Environment**

1. Create a Python virtual environment:
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App Locally

1. To run the app on your local machine:

```bash
python app.py
```

2. Access the app on localhost:5000