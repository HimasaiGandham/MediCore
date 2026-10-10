# 🧭 MediCore — Medical Question Intent Detection
> **Branch:** `feature/intent-detection`  
> **Role in MediCore:** Figuring out what kind of medical question the user is asking.

---

### 🌟 What is this branch all about?
Not all medical questions are the same! A person asking about an emergency needs a very different response than a student studying the history of diabetes. This branch provides **Intent Detection** 🧭 to understand the purpose of every question.

---

### 💡 Why do we need this in MediCore?
Consider these 3 questions:
1. *"My friend is having severe chest pain and can't breathe!"* &rarr; **Emergency Intent!** 🚨 (Show urgent 911 warning first!)
2. *"What is the standard dosage for Metformin?"* &rarr; **Medication Intent!** 💊 (Highlight dosage & precautions)
3. *"Which WHO guideline covers hypertension?"* &rarr; **Catalog Lookup Intent!** 📚 (Show catalog details)

Intent detection ensures each question gets the right tone, urgency, and layout!

---

### ⚙️ How does it work in simple words?
1. **Analyze the Question:** Inspects key words and question structure 🔍.
2. **Classify Intent:** Categorizes the inquiry (Emergency, Guideline Lookup, Treatment, Definition, or General Inquiry) 🏷️.
3. **Route Appropriately:** Directs the query through specialized pipelines (e.g., adding emergency banners for critical symptoms) 🚦.

---

### 🤝 How this branch contributes to MediCore
- 🚨 **Life-Saving Prioritization:** Immediately flags emergency situations and advises urgent medical help.
- 🎯 **Tailored Responses:** Formats answers according to what the user needs.
- ⚡ **Smarter Routing:** Skips unnecessary steps for simple catalog lookups.

---

### 🦄 What makes this branch unique?
This branch acts as the **triage nurse of MediCore**. It listens to the user's intent before deciding how the system should handle the question.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
