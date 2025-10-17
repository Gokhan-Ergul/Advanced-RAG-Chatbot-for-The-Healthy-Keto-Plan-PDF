# Advanced RAG Chatbot for "The Healthy Keto Plan" PDF

This project implements an advanced Retrieval-Augmented Generation (RAG) system to create an intelligent chatbot that answers questions based on the content of the book "The Healthy Keto Plan" by Dr. Eric Berg.

The notebook explores sophisticated retrieval techniques to ensure the answers are accurate, contextually relevant, and factually grounded in the source document.

## ✨ Key Features

This is not just a basic RAG implementation. It incorporates several advanced strategies to enhance performance:

-   **Hybrid Search**: Combines dense semantic search (via FAISS) with sparse keyword search (via BM25) to get the best of both worlds—understanding meaning while also catching specific keywords.
-   **Multi-Query Retriever**: Uses an LLM to automatically re-write a user's question into several different variations, broadening the search to capture a wider range of relevant documents.
-   **Conversational Memory**: Remembers the last few turns of the conversation, allowing for natural follow-up questions.
-   **Quantitative Evaluation**: Uses the **RAGAs** framework to scientifically measure the pipeline's performance on key metrics like faithfulness, answer relevancy, and context recall.

## 🛠️ Technology Stack

-   **LLM**: Google Gemini 2.5 Flash
-   **Framework**: LangChain
-   **Embeddings**: `sentence-transformers/all-mpnet-base-v2`
-   **Vector Store**: FAISS (Facebook AI Similarity Search)
-   **Data Processing**: `PyMuPDF (fitz)`, `pandas`
-   **Evaluation**: `RAGAs`

## ⚙️ Project Workflow

The Jupyter Notebook (`Keto.ipynb`) is structured in the following steps:

1.  **Data Loading & Preprocessing**: The PDF is downloaded, and text is extracted from each page. Initial cleaning is performed to remove artifacts.
2.  **Strategic Chunking**: A hybrid chunking strategy is used. The document is split into two parts:
    -   **Q&A Section**: Parsed with regex to create perfect, self-contained question-answer chunks.
    -   **Normal Prose**: Processed with `RecursiveCharacterTextSplitter` to maintain semantic context.
3.  **Embedding & Vectorization**: All text chunks are converted into vector embeddings and stored in a high-performance FAISS vector store.
4.  **Advanced Retrieval Setup**: An `EnsembleRetriever` is configured for hybrid search, which is then wrapped with a `MultiQueryRetriever` to enhance query understanding.
5.  **QA Chain Creation**: A `ConversationalRetrievalChain` is built, integrating the advanced retriever, conversational memory, and a custom prompt to guide the LLM's responses.
6.  **Evaluation**: A predefined set of 23 questions is run through the system, and the results are quantitatively evaluated using the RAGAs framework.

## 🚀 How to Run This Project

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/Advanced-RAG-Chatbot-for-The-Healthy-Keto-Plan-PDF.git](https://github.com/your-username/Advanced-RAG-Chatbot-for-The-Healthy-Keto-Plan-PDF.git)
cd Advanced-RAG-Chatbot-for-The-Healthy-Keto-Plan-PDF
```

### 2. Install Dependencies

It's recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install all the required packages:

```bash
pip install jupyter requests PyMuPDF pandas tqdm langchain langchain-google-genai sentence-transformers faiss-gpu rank_bm25 datasets ragas
```
*(Note: If you don't have an NVIDIA GPU, use `faiss-cpu` instead of `faiss-gpu`)*

### 3. Set Up Your API Key

You need a Google API key for the Gemini LLM.

-   Open the `Keto.ipynb` notebook.
-   Find the cell containing `genai.configure(api_key="...")`.
-   Replace the placeholder key with your actual Google API key.

### 4. Run the Jupyter Notebook

Launch Jupyter and open the notebook:

```bash
jupyter notebook Keto.ipynb
```

You can now run the cells sequentially to see the entire process from data loading to evaluation.

## 📊 Evaluation Results

The RAG pipeline was evaluated using the RAGAs framework on a set of 23 question-answer pairs. The results are excellent and demonstrate the effectiveness of the advanced retrieval strategies.

| Metric              | Score  | Description                                                                                             |
| ------------------- | :----: | ------------------------------------------------------------------------------------------------------- |
| **`faithfulness`** | 0.9596 | **Excellent.** The answers are factually consistent with the retrieved context, with almost no hallucination. |
| **`answer_relevancy`** | 0.8328 | **Very Good.** The answers are highly relevant and directly address the user's questions.               |
| **`context_recall`** | 1.0000 | **Perfect.** The retriever successfully found *all* the necessary information to answer the questions. |
| **`context_precision`** | `nan`  | **Not Calculated.** The score was `nan` due to API rate-limiting during evaluation, which prevented the metric from being properly computed. This is not an indicator of poor performance. |

## 🔮 Future Work

This project provides a strong foundation. The next steps for improvement are:

-   **Implement a Re-ranker**: Add a Cross-Encoder model (e.g., `BAAI/bge-reranker-base`) to re-rank the retrieved documents. This will improve `context_precision` by ensuring only the most relevant chunks are sent to the LLM.
-   **Add Metadata for Source Citing**: Enhance the chunking process to include page numbers as metadata, allowing the chatbot to cite the exact source of its information.
-   **Address API Rate Limits**: For more extensive evaluation, use a paid Google API plan or implement more robust backoff/retry logic to handle rate limits.
