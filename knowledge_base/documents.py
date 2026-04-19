"""
knowledge_base/documents.py
12 evidence-based medical documents.
Add or replace documents here; vectorstore.py will rebuild automatically.
"""

DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Chest Pain — Differential Diagnosis",
        "text": """Chest pain is one of the most common emergency presentations. A systematic approach to differential diagnosis is essential.

CARDIAC CAUSES:
Acute Coronary Syndrome (ACS): Includes unstable angina, NSTEMI, and STEMI. Typically presents as crushing central or left-sided chest pain radiating to left arm or jaw, diaphoresis, nausea, and dyspnoea. Risk factors include hypertension, diabetes, dyslipidaemia, smoking, and family history. ECG may show ST elevation (STEMI), ST depression or T-wave inversion (NSTEMI/UA), or be normal. Troponin I or T is the key biomarker — rises 3-6 hours after onset, peaks at 12-24 hours.
Pericarditis: Sharp pleuritic chest pain, worse lying flat, relieved by leaning forward. Friction rub on auscultation. ECG shows widespread saddle-shaped ST elevation.
Aortic Dissection: Sudden tearing or ripping pain radiating to the back; unequal blood pressure in both arms. CT aortography is diagnostic.

PULMONARY CAUSES:
Pulmonary Embolism (PE): Pleuritic chest pain, dyspnoea, haemoptysis, tachycardia. Risk factors: immobility, surgery, malignancy, thrombophilia. D-dimer useful for exclusion; CTPA is gold standard.
Pneumothorax: Sudden onset pleuritic pain and dyspnoea, especially in tall young men or COPD patients. Decreased breath sounds on affected side. CXR shows absent lung markings.
Pneumonia: Pleuritic pain with productive cough, fever, consolidation on CXR.

OTHER CAUSES:
GERD/Oesophageal: Burning retrosternal pain, worse after meals and lying down, relieved by antacids. No radiation to arm.
Musculoskeletal: Reproducible with palpation or movement, no radiation, no associated autonomic features.
Anxiety/Panic Disorder: Often associated with hyperventilation, palpitations, sense of doom, normal investigations.

IMMEDIATE RED FLAGS: ST elevation on ECG, haemodynamic instability, oxygen saturations below 90%, signs of aortic dissection."""
    },
    {
        "id": "doc_002",
        "topic": "Pneumonia — Diagnosis and CURB-65 Severity Score",
        "text": """Community-Acquired Pneumonia (CAP) is a lower respiratory tract infection acquired outside hospital.

CLINICAL FEATURES: Productive cough (purulent or rusty sputum), fever (>38°C), pleuritic chest pain, dyspnoea, tachypnoea, tachycardia. Examination: dullness to percussion, bronchial breathing, increased tactile vocal fremitus, crackles over the affected lobe.

INVESTIGATIONS:
- CXR: Consolidation (lobar or patchy), air bronchograms
- FBC: Leukocytosis (WBC >11 × 10⁹/L), neutrophilia
- CRP: Elevated; useful for monitoring response
- Sputum culture and sensitivity before antibiotics
- Blood cultures if severe
- Pneumococcal and Legionella urinary antigen in severe cases
- Arterial blood gas: assess for respiratory failure

CURB-65 SEVERITY SCORE — one point for each:
C — Confusion (new onset, AMTS ≤8)
U — Urea >7 mmol/L
R — Respiratory rate ≥30/min
B — Blood pressure: systolic <90 mmHg OR diastolic ≤60 mmHg
65 — Age ≥65 years

SCORE INTERPRETATION:
- 0-1: Low severity — consider home treatment
- 2: Moderate severity — short inpatient admission or close outpatient follow-up
- 3-5: High severity — hospitalisation required; consider ITU if score ≥4

TREATMENT: Amoxicillin 500mg TDS orally for mild CAP. Amoxicillin + clarithromycin for moderate. IV co-amoxiclav + IV clarithromycin for severe. Duration: 5 days for mild/moderate; 7-10 days for severe. Consider atypical organisms (Mycoplasma, Legionella, Chlamydophila) in younger patients or those not responding."""
    },
    {
        "id": "doc_003",
        "topic": "Pulmonary Embolism — Diagnosis and Wells Score",
        "text": """Pulmonary Embolism (PE) occurs when a thrombus (usually from DVT) embolises to the pulmonary vasculature.

CLINICAL FEATURES: Dyspnoea (most common), pleuritic chest pain, haemoptysis, tachycardia, tachypnoea, hypoxia, calf pain/swelling (DVT). Massive PE may present with syncope, hypotension, or cardiac arrest.

WELLS SCORE FOR PE — pre-test probability:
- Medical signs/symptoms of DVT: +3
- PE more likely than alternative diagnosis: +3
- Heart rate >100 bpm: +1.5
- Immobilisation ≥3 days OR surgery in last 4 weeks: +1.5
- Previous DVT or PE: +1.5
- Haemoptysis: +1
- Malignancy (on treatment, treated in last 6 months, or palliative): +1

INTERPRETATION:
- ≤4: Low probability — use D-dimer. If negative, PE excluded.
- >4: High probability — proceed to CTPA without waiting for D-dimer.

INVESTIGATIONS:
- D-dimer: Sensitive but not specific; elevated in many conditions. Use only in low/intermediate probability.
- CTPA (CT Pulmonary Angiography): Gold standard imaging.
- ECG: Sinus tachycardia most common. Classic S1Q3T3 pattern is rare but specific.
- CXR: Often normal; may show Hampton's hump or Westermark sign.
- ABG: Type 1 respiratory failure (low PaO₂, low PaCO₂).
- Echocardiography: Right heart strain in massive PE.

MANAGEMENT:
- Anticoagulation: LMWH (e.g., enoxaparin) immediately then transition to DOAC (apixaban/rivaroxaban) or warfarin.
- Thrombolysis (alteplase): Reserved for massive PE with haemodynamic compromise.
- Duration of anticoagulation: 3 months for provoked, at least 6 months for unprovoked or recurrent PE."""
    },
    {
        "id": "doc_004",
        "topic": "Acute Coronary Syndrome — STEMI and NSTEMI Management",
        "text": """Acute Coronary Syndrome (ACS) encompasses unstable angina, NSTEMI, and STEMI — a spectrum of myocardial ischaemia due to coronary artery disease.

CLINICAL PRESENTATION: Central/left-sided chest pain, pressure/crushing in nature, radiating to left arm, jaw or back. Associated diaphoresis, nausea, vomiting, dyspnoea. May be atypical in women, diabetics, and elderly (epigastric pain, fatigue, dyspnoea only).

ECG CHANGES:
- STEMI: ST elevation ≥1 mm in 2 contiguous limb leads or ≥2 mm in precordial leads. New LBBB.
- NSTEMI/UA: ST depression, T-wave inversion, or normal ECG.
- Posterior MI: ST depression V1-V3 with dominant R waves.

BIOMARKERS:
- Troponin I or T: Preferred marker. Rises 3-6 hours, peaks 12-24 hours, normalises 5-14 days. High-sensitivity troponin allows earlier rule-in/rule-out (0h/1h or 0h/2h pathways).
- CK-MB: Faster normalisation than troponin; useful for detecting reinfarction.

IMMEDIATE MANAGEMENT (MONA-B for STEMI):
- Morphine: Titrate for pain relief
- Oxygen: Only if SpO₂ <94%
- Nitrates: GTN spray/tablet if systolic BP >90 mmHg
- Aspirin: 300 mg loading dose
- P2Y12 inhibitor (ticagrelor 180 mg or clopidogrel 300 mg)

REPERFUSION (STEMI):
- Primary PCI: Gold standard if available within 120 minutes of first medical contact.
- Thrombolysis: If primary PCI not available within 120 minutes.

NSTEMI MANAGEMENT: Risk stratification using GRACE or TIMI score. Conservative vs invasive strategy based on risk. UFH or LMWH anticoagulation."""
    },
    {
        "id": "doc_005",
        "topic": "Diabetes Mellitus — Diagnosis, Types, and Management",
        "text": """Diabetes mellitus is a metabolic disorder characterised by chronic hyperglycaemia due to defects in insulin secretion, insulin action, or both.

DIAGNOSTIC CRITERIA (WHO 2006, confirmed on 2 separate occasions unless symptomatic):
- Fasting plasma glucose ≥7.0 mmol/L (126 mg/dL)
- 2-hour plasma glucose ≥11.1 mmol/L during OGTT
- Random plasma glucose ≥11.1 mmol/L with symptoms
- HbA1c ≥48 mmol/mol (6.5%)

TYPES:
Type 1 DM: Autoimmune destruction of pancreatic beta cells. Typically presents in younger patients, lean, with DKA. Absolute insulin deficiency. C-peptide low or absent. Positive autoantibodies (GAD, IA2, islet cell).
Type 2 DM: Insulin resistance with progressive beta cell failure. Usually older, overweight/obese. Insidious onset. Family history common.
MODY (Maturity-Onset Diabetes of the Young): Monogenic, autosomal dominant, young age of onset without autoantibodies or obesity. Consider if strong family history.
Secondary DM: Pancreatitis, Cushing's syndrome, acromegaly, haemochromatosis, drug-induced (steroids, thiazides).

MANAGEMENT:
Type 1: Multiple daily insulin injections or CSII (pump). Carbohydrate counting. Continuous glucose monitoring.
Type 2: Lifestyle modification first. First-line: Metformin. Second-line based on comorbidities:
  - ASCVD/Heart failure: SGLT-2 inhibitor (empagliflozin) or GLP-1 RA
  - Obesity: GLP-1 RA (semaglutide), SGLT-2 inhibitor
  - CKD: SGLT-2 inhibitor preferred

COMPLICATIONS: Microvascular (retinopathy, nephropathy, neuropathy), macrovascular (MI, stroke, peripheral arterial disease), foot complications, hypoglycaemia.
HbA1c target: Generally <53 mmol/mol (7%) but individualise based on patient."""
    },
    {
        "id": "doc_006",
        "topic": "Hypertension — Classification and Management",
        "text": """Hypertension (HTN) is defined as persistently elevated blood pressure (BP) ≥140/90 mmHg in medic, or ≥135/85 mmHg on ambulatory or home monitoring.

CLASSIFICATION (ESC/NICE):
- Normal: <130/85 mmHg
- High normal: 130-139/85-89 mmHg
- Stage 1 HTN: 140-159/90-99 mmHg (ABPM 135-149/85-94)
- Stage 2 HTN: ≥160/100 mmHg (ABPM ≥150/95)
- Hypertensive crisis: >180/120 mmHg
- Hypertensive emergency: >180/120 with end-organ damage

INVESTIGATIONS (to exclude secondary causes and assess end-organ damage):
- Urinalysis (proteinuria, haematuria), renal function and electrolytes
- Fasting glucose and lipids
- ECG (LVH, signs of previous MI)
- Fundoscopy (hypertensive retinopathy)
- ABPM/HBPM to confirm diagnosis
- Consider secondary causes if: young, resistant HTN, hypokalaemia (Conn's), renal bruit (RAS)

MANAGEMENT (NICE 2019 NG136):
Step 1: ACE inhibitor (or ARB if ACEi-intolerant) for <55 years or diabetics. Calcium channel blocker (CCB) for ≥55 years or Afro-Caribbean patients.
Step 2: ACEi/ARB + CCB
Step 3: ACEi/ARB + CCB + thiazide-like diuretic (chlortalidone/indapamide)
Step 4 (resistant HTN): Add spironolactone if K+ ≤4.5 mmol/L; or alpha-blocker or beta-blocker

LIFESTYLE: Weight reduction, DASH diet, reduce salt (<6g/day), alcohol limit, aerobic exercise 150 min/week, smoking cessation.
BP TARGETS: <140/90 mmHg in medic for most; <130/80 for diabetics and high-risk patients. <150/90 for those ≥80 years."""
    },
    {
        "id": "doc_007",
        "topic": "Anaemia — Types, Investigation, and Treatment",
        "text": """Anaemia is defined as haemoglobin below the lower limit of normal: <130 g/L in adult males, <120 g/L in adult females (non-pregnant).

CLASSIFICATION BY MCV:

MICROCYTIC (MCV <80 fL):
1. Iron deficiency anaemia (IDA): Most common. Low serum ferritin (<15 mcg/L), low serum iron, raised TIBC. Peripheral blood film: hypochromic microcytes, pencil cells, anisocytosis. Causes: blood loss (GI, menstrual), malabsorption (coeliac), poor dietary intake.
2. Thalassaemia: Target cells, raised HbA2 (beta-thal), family history, Mediterranean/South Asian origin. Ferritin normal or raised.
3. Anaemia of chronic disease (can be normocytic or microcytic): Low iron, low TIBC, raised ferritin. Associated with chronic infection, inflammation, or malignancy.
4. Sideroblastic anaemia: Ring sideroblasts on bone marrow, raised serum iron.

NORMOCYTIC (MCV 80-100 fL):
- Acute blood loss
- Haemolytic anaemia: Raised bilirubin, raised LDH, reduced haptoglobin, positive direct Coombs test (AIHA)
- Anaemia of chronic disease
- Aplastic anaemia, bone marrow infiltration

MACROCYTIC (MCV >100 fL):
- B12 deficiency: Neurological features (subacute combined degeneration of cord), glossitis, low serum B12, raised MMA. Causes: pernicious anaemia, veganism, malabsorption.
- Folate deficiency: No neurological features. Low red cell folate. Associated with poor diet, methotrexate, pregnancy, alcohol.
- Alcohol, hypothyroidism, liver disease, medications (hydroxyurea, azathioprine).

KEY INVESTIGATIONS: FBC, reticulocyte count, blood film, serum iron/TIBC/ferritin, B12, folate, haemolytic screen (LDH, haptoglobin, bilirubin, DAT), haemoglobin electrophoresis.
TREATMENT: Address underlying cause. Oral iron for IDA (ferrous sulphate 200mg TDS). B12 IM injections for pernicious anaemia. Folate 5mg orally for folate deficiency."""
    },
    {
        "id": "doc_008",
        "topic": "Thyroid Disorders — Hypothyroidism and Hyperthyroidism",
        "text": """The thyroid gland produces thyroxine (T4) and triiodothyronine (T3) under TSH stimulation from the anterior pituitary.

HYPOTHYROIDISM:
Causes: Hashimoto's thyroiditis (autoimmune, anti-TPO antibodies, most common in the West), post-radioiodine therapy, thyroidectomy, secondary (pituitary failure), drugs (amiodarone, lithium).
Symptoms: Fatigue, weight gain, cold intolerance, constipation, dry skin, hair loss, bradycardia, slow-relaxing reflexes, myxoedema, depression, cognitive slowing, menorrhagia, dyslipidaemia.
Investigations: TSH elevated, free T4 low (overt). TSH elevated, T4 normal (submedical). Anti-TPO antibodies.
Treatment: Levothyroxine, starting dose 1.6 mcg/kg/day. Recheck TSH after 4-8 weeks. Target TSH: 0.4-2.5 mIU/L.

HYPERTHYROIDISM:
Causes: Graves' disease (most common; TSH-receptor antibodies/TRAb), toxic multinodular goitre, toxic adenoma (Plummer's disease), subacute thyroiditis, excessive levothyroxine.
Symptoms: Weight loss despite increased appetite, heat intolerance, sweating, tremor, palpitations (AF risk), anxiety, diarrhoea, proximal myopathy, lid lag, exophthalmos (Graves' specific).
Investigations: TSH suppressed (<0.05), free T4 and/or T3 elevated. TRAb positive in Graves'. Thyroid uptake scan.
Treatment:
  - Antithyroid drugs: Carbimazole (block thyroid synthesis) or propylthiouracil (pregnancy — 1st trimester). Block-and-replace or titration regimen.
  - Beta-blockers (propranolol): Symptom control, especially for tremor and palpitations.
  - Radioiodine (I-131): Definitive, leads to hypothyroidism in most.
  - Thyroidectomy: Preferred if large goitre, suspicious nodule, or patient preference.
THYROID STORM: Life-threatening. High fever, severe tachycardia, agitation, confusion. Treat with PTU, Lugol's iodine, propranolol, dexamethasone, ICU."""
    },
    {
        "id": "doc_009",
        "topic": "Stroke — Recognition, Types, and Emergency Management",
        "text": """A stroke is the sudden onset of focal neurological deficit lasting >24 hours (or any duration with imaging evidence), caused by cerebrovascular pathology.

FACE-ARM-SPEECH TEST (FAST): Facial drooping, arm weakness, speech problems, Time to call 999.

TYPES:
Ischaemic (85%): Atherothrombotic (large vessel), cardioembolic (AF, valvular disease), lacunar (small vessel), or cryptogenic. Presents with sudden focal deficit: hemiplegia, hemisensory loss, hemianopia, aphasia, dysarthria, neglect.
Haemorrhagic (15%): Intracerebral haemorrhage (HTN, amyloid angiopathy) or subarachnoid haemorrhage (ruptured aneurysm). SAH: thunderclap headache — worst headache of life, meningism, photophobia.

BAMFORD/OXFORDSHIRE CLASSIFICATION (ischaemic):
- TACS (Total Anterior Circulation Stroke): Motor/sensory deficit, hemianopia, higher cortical dysfunction (all three).
- PACS (Partial ACS): 2 of 3 features above, or isolated cortical feature.
- POCS (Posterior Circulation): Cerebellar, brainstem, or occipital features.
- LACS (Lacunar): Pure motor, pure sensory, ataxic hemiparesis, sensorimotor — no cortical features.

EMERGENCY MANAGEMENT:
1. Non-contrast CT head immediately to exclude haemorrhage.
2. If ischaemic and within 4.5 hours of onset: IV alteplase (thrombolysis) — check contraindications.
3. If large vessel occlusion: Mechanical thrombectomy up to 24 hours in selected patients.
4. Blood pressure management: Target <185/110 pre-thrombolysis; <180/105 post.
5. Antiplatelet therapy: Aspirin 300 mg stat (after haemorrhage excluded); start dual antiplatelet.
6. Admit to dedicated stroke unit.
7. ABCD2 score for TIA — to stratify 2-day stroke risk and guide urgency of investigation."""
    },
    {
        "id": "doc_010",
        "topic": "Sepsis — Recognition, Sepsis-3 Definition, and Sepsis Six Bundle",
        "text": """Sepsis is defined (Sepsis-3, 2016) as life-threatening organ dysfunction caused by a dysregulated host response to infection.

SEPSIS-3 CRITERIA:
Suspected infection PLUS acute organ dysfunction with a SOFA score increase of ≥2 from baseline.

qSOFA (quick SOFA — bedside screening):
- Respiratory rate ≥22/min (+1)
- Altered mentation: GCS <15 (+1)
- Systolic BP ≤100 mmHg (+1)
qSOFA ≥2 is associated with poor outcomes — escalate care.

SEPTIC SHOCK: Sepsis with MAP <65 mmHg requiring vasopressors AND serum lactate >2 mmol/L despite adequate fluid resuscitation. Mortality >40%.

SEPSIS SIX (Hour-1 Bundle):
1. Administer high-flow oxygen
2. Take blood cultures (2 sets) BEFORE antibiotics
3. Give IV broad-spectrum antibiotics within 1 hour
4. Give IV crystalloid fluid: 30 mL/kg stat for hypotension or lactate ≥4 mmol/L
5. Measure lactate
6. Measure urine output (catheterise if needed)

ANTIBIOTIC CHOICE: Guided by suspected source. Broad-spectrum: piperacillin-tazobactam ± gentamicin, or meropenem if high-risk. De-escalate based on cultures. Duration typically 5-7 days depending on source control.
MONITORING: Vasopressors (noradrenaline first-line) if MAP <65 despite fluids. ICU admission for septic shock."""
    },
    {
        "id": "doc_011",
        "topic": "Acute Appendicitis — Diagnosis and Alvarado Score",
        "text": """Acute appendicitis is the most common surgical emergency, caused by obstruction of the appendix lumen with subsequent infection and possible perforation.

CLINICAL FEATURES: Periumbilical pain migrating to right iliac fossa (RIF) over 12-24 hours. Anorexia, nausea, vomiting, low-grade fever (37.5-38.5°C). RIF tenderness at McBurney's point.
Examination signs:
- Rovsing's sign: Palpation of LIF causes pain in RIF
- Psoas sign: Extension of right hip causes pain (retrocaecal appendix)
- Obturator sign: Internal rotation of right hip causes pain (pelvic appendix)
- Rebound tenderness: Peritoneal irritation

ALVARADO SCORE (MANTRELS):
M — Migration of pain to RIF (+1), A — Anorexia (+1), N — Nausea/Vomiting (+1), T — RIF tenderness (+2), R — Rebound tenderness (+1), E — Elevated temperature >37.3°C (+1), L — Leucocytosis (+2), S — Shift to neutrophils (+1).
Score ≥7 suggests appendicitis.

INVESTIGATIONS:
- FBC: Leukocytosis with neutrophilia, CRP raised
- Urinalysis: To exclude UTI/renal colic
- Pregnancy test (women of childbearing age) — to exclude ectopic pregnancy
- USS abdomen: First-line in children, young women.
- CT abdomen/pelvis: Gold standard if diagnosis uncertain — >95% sensitivity/specificity

MANAGEMENT: IV fluids, NBM, analgesia. Laparoscopic appendicectomy: Treatment of choice. Pre-operative: IV co-amoxiclav or metronidazole + cephalosporin. Continue for 5 days if perforated."""
    },
    {
        "id": "doc_012",
        "topic": "Heart Failure — Classification, Diagnosis, and Management",
        "text": """Heart failure (HF) is a medical syndrome in which the heart fails to pump sufficient blood to meet metabolic demands, or does so only at elevated filling pressures.

CLASSIFICATION:
- HFrEF (reduced EF): EF <40%. Dilated ventricle, impaired systolic function.
- HFmrEF (mildly reduced EF): EF 40-49%.
- HFpEF (preserved EF): EF ≥50%. Impaired diastolic function. Often associated with HTN, AF, obesity, diabetes.
NYHA Classification: I (no symptoms) → IV (symptoms at rest).

CLINICAL FEATURES:
Left heart failure: Dyspnoea (exertional → orthopnoea → PND), fatigue, pulmonary oedema, fine bibasal crackles, frothy/pink sputum.
Right heart failure: Peripheral oedema (pitting, sacral), raised JVP, hepatomegaly, ascites.
Both: Third heart sound (gallop), displaced apex beat.

INVESTIGATIONS:
- BNP/NT-proBNP: Primary diagnostic biomarker. BNP >100 pg/mL or NT-proBNP >300 pg/mL supports HF.
- ECG: AF, LVH, previous MI, LBBB
- CXR: Cardiomegaly (CTR >0.5), upper lobe venous diversion, Kerley B lines, bilateral perihilar shadowing (bat wing), pleural effusions
- Echocardiogram: Gold standard — confirms EF, identifies aetiology

MANAGEMENT OF HFrEF (the Fantastic Four disease-modifying drugs):
1. ACE inhibitor/ARB/ARNI (sacubitril-valsartan): Reduce mortality
2. Beta-blocker (bisoprolol, carvedilol): Reduce mortality
3. MRA (spironolactone/eplerenone): Reduce mortality
4. SGLT-2 inhibitor (dapagliflozin/empagliflozin): Reduce hospitalisations
Diuretics (furosemide) for symptom relief — not disease-modifying."""
    },
]
