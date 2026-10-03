🩺 MediCore – Healthcare Information Assistant
MediCore is a healthcare-focused project that helps users find relevant information from medical documents without having to manually search through long PDFs.
Instead of reading through hundreds of pages, users can simply ask a question, and MediCore finds the relevant information and provides an answer along with its source and page details.
🚀 Project Overview
The main idea behind MediCore is simple:
Ask a question → MediCore finds the relevant information → The system generates an answer with its source.
For example:
User Question	MediCore Response
What are the criteria for Stage 2 hypertension?	Relevant information from the medical document
What are the glycemic targets for Type 2 diabetes?	Relevant information with source details
What is the treatment for appendicitis?	Information not available in the provided documents


If the required information is not available, MediCore avoids generating an unsupported answer.
✨ Features
- 📄 Medical document processing
- 🔎 Relevant information retrieval
- 🧠 AI-generated answers based on available documents
- 📑 Source and page information
- 🚫 Handling of unsupported questions
- 📊 Retrieval evaluation
- 💬 Simple and easy-to-use interface
- ⚡ Quick access to information from lengthy documents
🛠️ Technologies Used
- Python – Main development language
- PyMuPDF – Medical PDF processing
- Sentence Transformers – Converting text into meaningful representations
- FAISS – Finding relevant information from the documents
- Gemini – Answer generation
- Streamlit – User interface
🎯 Project Objective
The objective of MediCore is to make medical information easier and faster to find from lengthy reference documents.
The project demonstrates the practical use of:
- Retrieval-Augmented Generation
- Natural Language Processing
- Semantic Search
- Document Processing
- Artificial Intelligence
- Information Retrieval
💡 Why MediCore?
Sometimes even a simple health-related doubt can make us search through long articles, websites, and medical PDFs.
MediCore explores a simpler approach by allowing users to ask their question directly and find the relevant information from the available medical documents.
It also shows where the information came from, making the answer easier to verify.
📊 Evaluation
MediCore was tested using questions based on the available medical documents.
- Top-1 Retrieval Accuracy: 100%
- Top-3 Retrieval Accuracy: 100%
- Unsupported questions were also tested to ensure the system does not provide information that is not present in the documents.
👨‍💻 Project Status
✅ Main System Completed
MediCore can currently process medical documents, retrieve relevant information, generate answers using the retrieved content, display source details, handle unsupported questions, and provide a simple user interface.
Future improvements include adding more medical documents, improving complex-question retrieval, and expanding the application.
🔮 Future Scope
- 📚 Add more medical documents
- 📄 Support additional document formats
- 🔎 Improve retrieval for complex questions
- 💬 Add conversation history
- 🎨 Improve the user interface
- 🌐 Make the application available online
📖 Learning Reference
The basic RAG concept was learned from LangChain's RAG From Scratch project.
The concept was then applied to build our own healthcare-focused project, MediCore.
📄 Disclaimer
MediCore is an academic project created for educational and informational purposes.
It is not intended to provide medical diagnosis, treatment, or professional medical advice.
For medical concerns, users should consult a qualified healthcare professional.
⭐ MediCore
Ask your question. Find the information. Know the source.
