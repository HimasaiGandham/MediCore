# 🎯 MediCore — Grounded Clinical Prompts & Anti-Hallucination
> **Branch:** `feature/grounded-response`  
> **Role in MediCore:** Enforcing strict truth guardrails so the AI never makes up facts.

---

### 🌟 What is this branch all about?
General-purpose AI chatbots can sometimes "hallucinate" — invent convincing-sounding facts that are completely wrong 🚨. In healthcare, hallucination is dangerous! This branch builds **Grounded Prompts** 🎯 that force the AI to speak ONLY from approved medical text.

---

### 💡 Why do we need this in MediCore?
If a patient asks about medication dosage, an unconstrained AI might guess a number from memory.

With grounded prompts:
- The AI is given the exact retrieved excerpt from the CDC or WHO 📄.
- The prompt strictly instructs: *"Only answer using the text above. If the text does not contain the answer, say you do not know. Never guess."* 🛑
- The AI becomes a faithful reader rather than a creative writer!

---

### ⚙️ How does it work in simple words?
1. **Build Grounded Prompt:** Wraps user questions and retrieved evidence in strict system instructions 🧱.
2. **Context Anchoring:** Feeds the exact guideline title, section, and text directly into the AI prompt ⚓.
3. **Zero Fabrication:** The AI synthesizes the answer directly from the provided evidence and explicitly cites where it found each detail 📑.

---

### 🤝 How this branch contributes to MediCore
- 🛡️ **Prevents Medical Hallucinations:** Stops the AI from making up unverified medical advice.
- 🩺 **Faithful Synthesis:** Answers accurately reflect official guidelines from WHO, CDC, and NHS.
- 🎓 **High Scientific Integrity:** Crucial for medical education and clinical research.

---

### 🦄 What makes this branch unique?
This branch is the **truth enforcement officer of MediCore**. It writes the prompt engineering guardrails that keep the AI anchored in verified evidence.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
