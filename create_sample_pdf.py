"""
Helper script to generate a sample multi-page medical PDF document
for testing the Healthcare RAG ingestion pipeline.
"""
from pathlib import Path
import fitz

def generate_medical_pdf():
    project_root = Path(__file__).resolve().parent
    docs_dir = project_root / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = docs_dir / "medical_information.pdf"

    doc = fitz.open()

    # --- Page 1: Hypertension Clinical Overview ---
    p1 = doc.new_page(width=595, height=842)  # A4 size
    content_p1 = """CLINICAL PRACTICE GUIDELINE: HYPERTENSION MANAGEMENT
Document ID: MED-REF-2026-01 | Department of Internal Medicine

1. Diagnostic Criteria & Staging
- Normal Blood Pressure: Systolic < 120 mmHg and Diastolic < 80 mmHg.
- Elevated Blood Pressure: Systolic 120-129 mmHg and Diastolic < 80 mmHg.
- Stage 1 Hypertension: Systolic 130-139 mmHg or Diastolic 80-89 mmHg.
- Stage 2 Hypertension: Systolic >= 140 mmHg or Diastolic >= 90 mmHg.
- Hypertensive Crisis: Systolic > 180 mmHg and/or Diastolic > 120 mmHg.

2. Initial Non-Pharmacological Interventions
- Dietary Approaches to Stop Hypertension (DASH diet): High in vegetables, fruits, and whole grains.
- Sodium restriction: Target < 1,500 mg/day (or at least 1,000 mg/day reduction).
- Physical activity: 90-150 minutes of aerobic and dynamic resistance exercise per week.
- Moderation of alcohol intake and smoking cessation.

3. First-Line Pharmacotherapy
- Thiazide diuretics (e.g., Chlorthalidone 12.5-25 mg daily, Hydrochlorothiazide 25-50 mg daily).
- Angiotensin-Converting Enzyme (ACE) Inhibitors (e.g., Lisinopril 10-40 mg daily).
- Angiotensin Receptor Blockers (ARBs) (e.g., Losartan 50-100 mg daily).
- Calcium Channel Blockers (CCBs) (e.g., Amlodipine 5-10 mg daily).
Note: ACE inhibitors and ARBs should not be combined simultaneously due to renal risk.
"""
    p1.insert_text((50, 70), content_p1, fontsize=11, lineheight=1.4)

    # --- Page 2: Type 2 Diabetes Management ---
    p2 = doc.new_page(width=595, height=842)
    content_p2 = """CLINICAL PRACTICE GUIDELINE: TYPE 2 DIABETES MELLITUS
Document ID: MED-REF-2026-02 | Endocrinology Division

1. Glycemic Targets
- Target HbA1c for non-pregnant adults: < 7.0% (53 mmol/mol).
- Fasting and preprandial capillary plasma glucose: 80-130 mg/dL (4.4-7.2 mmol/L).
- Peak postprandial capillary plasma glucose: < 180 mg/dL (10.0 mmol/L).
- More relaxed targets (HbA1c < 8.0%) are appropriate for patients with severe hypoglycemia history.

2. Pharmacotherapy Pathway
- First-line therapy: Metformin (initial 500 mg twice daily with meals, titrating up to 2000 mg/day)
  combined with lifestyle intervention.
- In patients with established atherosclerotic cardiovascular disease (ASCVD), heart failure (HF),
  or chronic kidney disease (CKD):
  * SGLT2 Inhibitors (e.g., Empagliflozin 10-25 mg daily, Dapagliflozin 10 mg daily).
  * GLP-1 Receptor Agonists (e.g., Semaglutide 0.5-1.0 mg weekly, Liraglutide 1.2-1.8 mg daily).

3. Monitoring and Complication Screenings
- Annual comprehensive foot examination to evaluate neuropathy and peripheral vascular disease.
- Annual urine albumin-to-creatinine ratio (uACR) and estimated GFR.
- Dilated eye exam every 1-2 years by an optometrist or ophthalmologist.
"""
    p2.insert_text((50, 70), content_p2, fontsize=11, lineheight=1.4)

    # --- Page 3: Emergency Red Flags & Critical Protocols ---
    p3 = doc.new_page(width=595, height=842)
    content_p3 = """CRITICAL PROTOCOLS: EMERGENCY RED FLAGS & CONTRAINDICATIONS
Document ID: MED-REF-2026-03 | Emergency & Critical Care

1. Acute Coronary Syndrome (ACS) Red Flags
- Crushing substernal chest pressure radiating to left jaw, neck, or shoulder.
- Associated diaphoresis, dyspnea, nausea, and lightheadedness.
- Immediate actions: Oxygen if SpO2 < 90%, Chewable Aspirin 324 mg, Sublingual Nitroglycerin (if not contraindicated).
- Contraindications to Nitroglycerin: SBP < 90 mmHg, severe bradycardia, or PDE-5 inhibitor use within 24-48 hours.

2. Sepsis & Septic Shock (qSOFA Criteria)
- Respiratory rate >= 22 breaths per minute.
- Altered mental status (Glasgow Coma Scale < 15).
- Systolic blood pressure <= 100 mmHg.
- Management: Blood cultures prior to broad-spectrum IV antibiotics within 1 hour, 30 mL/kg crystalloid fluid bolus.

3. Anaphylaxis Emergency Action
- Sudden cutaneous symptoms (urticaria, angioedema) plus respiratory compromise or hypotension.
- Primary treatment: Epinephrine (1:1,000 / 1 mg/mL) 0.3-0.5 mg Intramuscularly into anterolateral thigh immediately.
- Never delay Epinephrine administration for secondary antihistamines or corticosteroids.
"""
    p3.insert_text((50, 70), content_p3, fontsize=11, lineheight=1.4)

    doc.save(str(pdf_path))
    doc.close()
    print(f"Sample medical PDF successfully created at: {pdf_path}")

if __name__ == "__main__":
    generate_medical_pdf()
