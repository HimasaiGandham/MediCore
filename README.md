# 🧠 MediCore — Multi-Turn Conversation Memory
> **Branch:** `feature/conversation-memory`  
> **Role in MediCore:** Remembering past chat messages so you can ask natural follow-up questions.

---

### 🌟 What is this branch all about?
When you talk to a human doctor, you don't repeat your entire life story with every sentence! This branch gives MediCore **Conversation Memory** 🧠💬 so it remembers what you were just talking about.

---

### 💡 Why do we need this in MediCore?
Consider this natural conversation:
- **User:** *"What are the first-line medications for Type 2 Diabetes?"*
- **MediCore:** *"According to CDC and WHO, Metformin is typically the first-line medication."*
- **User:** *"What are its common side effects?"*

Without memory, the system would ask: *"What medication are you talking about?"* 🤦‍♂️  
With conversation memory, MediCore knows that *"its"* refers to Metformin and answers seamlessly!

---

### ⚙️ How does it work in simple words?
1. **Session Diary:** The system logs recent user questions and assistant answers in a private chat history 📜.
2. **Context Window:** When you ask a follow-up question, MediCore reviews the past few messages to understand the topic 💡.
3. **Smart Reset:** Users can easily clear the chat history whenever they want to start a brand-new medical topic 🔄.

---

### 🤝 How this branch contributes to MediCore
- 🗣️ **Natural Conversations:** Feels like talking to an attentive human assistant.
- ⏱️ **Saves User Time:** No need to re-type long explanations in every follow-up query.
- 🏥 **Comprehensive Consultations:** Allows multi-step discussions exploring symptoms, treatments, and precautions.

---

### 🦄 What makes this branch unique?
This branch manages **dialogue state and conversational context**, ensuring that multi-turn consultations remain coherent and context-aware.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
