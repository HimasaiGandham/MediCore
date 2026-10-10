# 🏷️ MediCore — Medical Source & Authority Filtering
> **Branch:** `feature/medical-source-filter`  
> **Role in MediCore:** Letting users filter answers by specific health agencies (WHO, CDC, NHS, ICMR).

---

### 🌟 What is this branch all about?
Sometimes you don't want to search the entire world's medical guidelines at once — you only want guidance from a **specific trusted authority** 🏷️! This branch introduces **Source & Authority Filtering**.

---

### 💡 Why do we need this in MediCore?
- A doctor in the UK might specifically want guidance from the **NHS** 🇬🇧.
- An Indian medical student might want protocols from the **ICMR** 🇮🇳.
- A public health researcher might want global recommendations from the **WHO** 🌍.

This branch adds filters so users can narrow searches to specific agencies or medical topics (Cardiology, Emergency, Infectious Diseases)!

---

### ⚙️ How does it work in simple words?
1. **Select Filter:** In the web interface, the user selects an authority filter (e.g., "Only CDC") 🎛️.
2. **Smart Scoping:** The search engine limits its FAISS vector search only to chunks that match the selected filter 🎯.
3. **Targeted Results:** The answer is generated exclusively from the chosen authority's guidelines 📑.

---

### 🤝 How this branch contributes to MediCore
- 🎯 **Region-Specific Guidance:** Users can view guidelines relevant to their local health system.
- ⚡ **Faster, Focused Search:** Searches smaller, focused subsets of the knowledge base.
- 🩺 **Domain Isolation:** Easily compare how different organizations approach the same health condition.

---

### 🦄 What makes this branch unique?
This branch adds **metadata-driven filtering controls**, giving users the power to customize which medical authorities power their answers.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
