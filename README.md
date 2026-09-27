# MediCore 🩺

### Healthcare Document Question Answering using RAG

MediCore is a healthcare-focused **Retrieval-Augmented Generation (RAG)** project that allows users to ask questions about medical information stored in PDF documents.

Instead of directly asking an AI model to answer a question, MediCore first searches the uploaded medical documents and retrieves the most relevant information. This information is then given to Gemini to generate a grounded answer.

---

## 📌 What is MediCore?

MediCore helps users find information from medical documents using natural language questions.

For example:

**Question:**
> What are the criteria for Stage 2 hypertension?

MediCore searches the medical PDF, finds the relevant information, and generates an answer using that retrieved content.

It also shows the **source document, page number, chunk, and similarity score** used for the answer.

---

## 🔄 How MediCore Works

The system follows a simple RAG pipeline:

```text
Medical PDF
     ↓
Extract Text
     ↓
Split into Chunks
     ↓
Create Embeddings
     ↓
Store in FAISS
     ↓
User asks a Question
     ↓
Find Relevant Chunks
     ↓
Send Context to Gemini
     ↓
Generate Answer
     ↓
Show Answer + Sources
✨ Features
📄 Extracts text from medical PDF documents
✂️ Splits documents into smaller chunks
🧠 Creates embeddings using Sentence Transformers
🔎 Performs semantic search using FAISS
🤖 Generates answers using Gemini
📚 Shows the source document and page number
📊 Displays similarity scores
🚫 Handles questions that are not covered by the documents
🖥️ Provides a simple Streamlit web interface
🔐 Keeps the Gemini API key inside a .env file
🛠️ Technologies Used
Technology	Purpose
Python	Main programming language
PyMuPDF	Extract text from PDF
Sentence Transformers	Create text embeddings
FAISS	Semantic search
Gemini	Answer generation
Streamlit	Web interface
Git & GitHub	Version control
📂 Project Structure
MediCore/
│
├── documents/
│   └── medical_information.pdf
│
├── src/
│   ├── main.py
│   ├── evaluation.py
│   └── app.py
│
├── evaluation_results/
│   ├── results.json
│   └── evaluation_report.txt
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
⚙️ RAG Pipeline
1. Document Extraction

MediCore uses PyMuPDF to extract text from the medical PDF page by page.

2. Text Chunking

The extracted text is divided into smaller chunks so that relevant information can be retrieved efficiently.

The current system uses approximately:

Chunk size: 500 characters
Overlap: 50 characters

Each chunk also stores information such as:

Document name
Page number
Chunk ID
3. Embeddings

MediCore uses:

sentence-transformers/all-MiniLM-L6-v2

The model converts each text chunk into a 384-dimensional vector.

4. FAISS Retrieval

The embeddings are stored in a FAISS index.

When a user asks a question:

The question is converted into an embedding.
FAISS searches for similar chunks.
The top relevant chunks are retrieved.
These chunks are passed to Gemini.
5. Gemini Answer Generation

Gemini receives the user's question along with the retrieved medical context.

The prompt instructs Gemini to answer using the provided context instead of inventing information.

If the required information is not available, MediCore returns:

The available reference material does not contain sufficient information to answer this question.
🖥️ Web Interface

MediCore includes a Streamlit interface where users can:

Enter a medical question
View the generated answer
See the retrieved sources
View page numbers
View similarity scores
Inspect the retrieved text

To start the application:

cd D:\MediCore
.\.venv\Scripts\streamlit.exe run src\app.py

Then open:

http://localhost:8501
🔑 API Key Setup

MediCore uses the Gemini API for answer generation.

Create a .env file in the project folder:

LLM_PROVIDER=gemini
LLM_API_KEY=YOUR_GEMINI_API_KEY

The API key should not be written directly inside the Python code.

The .env file is included in .gitignore so that the API key is not uploaded to GitHub.

🧪 Example Questions

You can try questions such as:

What are the criteria for Stage 2 hypertension?
What are the glycemic targets for Type 2 diabetes?

You can also test unsupported questions such as:

What is the treatment for appendicitis?

If the information is not available in the reference document, the system should not generate an unsupported medical answer.

📊 Evaluation

MediCore was tested using a set of questions based on the available medical document.

The retrieval evaluation achieved:

Top-1 Retrieval Accuracy : 100%
Top-3 Retrieval Accuracy : 100%

Unsupported-question handling was also tested to make sure the system does not provide information that is not present in the reference document.

💡 What Makes This Project Different?

The basic RAG workflow was studied using LangChain's RAG From Scratch repository.

Based on that concept, MediCore was built specifically for healthcare documents.

The project focuses on:

Medical document retrieval
Semantic search
Grounded Gemini responses
Page-level source tracking
Similarity scores
Unsupported-question handling
Retrieval evaluation

Reference:

https://github.com/langchain-ai/rag-from-scratch

🎯 Main Challenge

One of the main challenges was connecting the Gemini API with the RAG pipeline.

The retrieved medical information had to be passed correctly to Gemini so that the generated answer was based on the available document.

During development, we also handled:

API authentication
API keys
API quota limitations
API errors
Unsupported questions
📚 What I Learned

Through this project, I learned how a RAG system works from start to finish.

I learned how to:

Extract information from PDFs
Split documents into chunks
Generate embeddings
Perform semantic search using FAISS
Connect retrieved information with Gemini
Use APIs securely
Handle API errors and limitations
Evaluate retrieval accuracy
Build a simple web interface using Streamlit

The project also helped me understand that an AI system should not only generate answers, but should also provide the information that supports those answers.

⚠️ Limitations

Currently, MediCore has some limitations:

The system can only answer questions related to the available medical documents.
It depends on the Gemini API for answer generation.
It does not provide real-time medical information.
The quality of retrieval depends on the available documents.
It is not designed for medical diagnosis or clinical decision-making.
The current document collection is limited.
🚀 Future Improvements

Some possible improvements are:

Add more medical documents
Support multiple PDF files
Improve retrieval for complex questions
Add hybrid search
Add a re-ranking system
Add chat history
Improve the evaluation dataset
Support more document formats
Deploy the application online
⚠️ Disclaimer

MediCore is an academic/project-based healthcare information retrieval system.

It is intended for retrieving information from the provided reference documents and is not a replacement for professional medical advice, diagnosis, or treatment.

👨‍💻 Project

Project Name: MediCore
Project Type: Healthcare RAG System
Language: Python
Interface: Streamlit

Medical Documents
       ↓
    MediCore
       ↓
Semantic Retrieval
       ↓
Grounded Answer
       ↓
Source Information
Built as an academic project to explore
Retrieval-Augmented Generation in healthcare.
