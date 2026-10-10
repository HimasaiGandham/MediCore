# 🗂️ MediCore — Central Medical Catalog & Document Manager
> **Branch:** `feature/document-manager`  
> **Role in MediCore:** Organizing, cataloging, and browsing the entire library of medical guidelines.

---

### 🌟 What is this branch all about?
When a medical system has dozens of guidelines on asthma, hypertension, diabetes, and stroke, you need a smart librarian to keep track of everything 🗂️. This branch provides the **Central Document Manager & Catalog** (`catalog.json`).

---

### 💡 Why do we need this in MediCore?
Without an organized catalog, files become a messy pile of documents. A doctor or researcher needs to quickly see:
- How many guidelines are currently indexed?
- Which guidelines come from the NHS vs WHO vs CDC?
- What year was each guideline updated?

This branch builds the central inventory that powers the catalog browser tab in the web application!

---

### ⚙️ How does it work in simple words?
1. **Central Registry:** Maintains `documents/catalog.json` with structured info for every document (ID, Title, Authority, Year, Category) 📋.
2. **Library Statistics:** Calculates live metrics like total indexed guidelines, total text chunks, and covered medical domains 📊.
3. **Catalog Browser:** Powers the "Medical Guidelines Catalog" tab in the UI so users can search and filter guidelines by agency 🔍.

---

### 🤝 How this branch contributes to MediCore
- 📚 **Total Transparency:** Users can see exactly which medical guidelines MediCore knows about.
- 🏷️ **Clear Taxonomy:** Groups guidelines into categories like Emergency, Cardiology, and Infectious Diseases.
- ⚡ **Easy System Management:** Makes adding, updating, or inspecting guidelines straightforward.

---

### 🦄 What makes this branch unique?
This branch is the **central registry and inventory manager**. It maintains the metadata schema that lets users explore the entire medical knowledge base.

---

### ⚠️ Important Health Reminder
MediCore is an educational and reference assistant. It does not replace the judgment of a licensed doctor. Always call your local emergency service in critical situations! 🚑
