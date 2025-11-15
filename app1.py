# app.py
import streamlit as st
import re
import os
import json
import requests
from time import sleep
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# Optional: import your Google/genai client only if you have keys; code handles absence.
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ============================
# CONFIG
# ============================
load_dotenv()
GENIE_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GENIE_API_KEY")
if genai and GENIE_KEY:
    try:
        genai.configure(api_key=GENIE_KEY)
    except Exception:
        pass

st.set_page_config(page_title="Agentic AI Healthcare Assistant (v3)", layout="wide", page_icon="🧬")

# ============================
# HELPERS / CACHING
# ============================
@st.cache_resource
def load_embedding_model():
    # lightweight model; change to a heavier one if desired
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

CANONICAL_SYMPTOMS = [
    "headache", "fever", "cough", "sore throat", "runny nose", "shortness of breath",
    "chest pain", "stomach ache", "nausea", "vomiting", "diarrhea",
    "back pain", "neck pain", "elbow pain", "knee pain", "wrist pain",
    "joint pain", "muscle pain", "fatigue", "dizziness", "rash",
    "itching", "anxiety", "depression", "weight loss", "loss of appetite",
    "vision changes", "weakness", "numbness"
]

CAN_EMB = embedding_model.encode(CANONICAL_SYMPTOMS, show_progress_bar=False)

SPECIALIST_MAP = {
    "elbow pain": "Orthopedic Specialist",
    "knee pain": "Orthopedic Specialist",
    "wrist pain": "Orthopedic Specialist",
    "joint pain": "Orthopedic Specialist",
    "back pain": "Orthopedic Specialist",
    "neck pain": "Orthopedic Specialist",
    "chest pain": "Cardiologist",
    "shortness of breath": "Pulmonologist",
    "stomach ache": "Gastroenterologist",
    "nausea": "Gastroenterologist",
    "vomiting": "Gastroenterologist",
    "rash": "Dermatologist",
    "itching": "Dermatologist",
    "headache": "Neurologist",
    "vision changes": "Neurologist",
    "depression": "Psychiatrist",
    "anxiety": "Psychiatrist",
    "default": "General Physician"
}

def extract_json_list_from_text(text):
    import re, json
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

# ============================
# SYMPTOM EXTRACTION
# ============================
def extract_symptoms_with_parts(user_text, min_sim=0.52, top_n=6):
    text = user_text.lower()
    body_parts = ["elbow", "knee", "shoulder", "wrist", "ankle", "back", "neck", "hip", "jaw"]
    explicit = []
    for bp in body_parts:
        if re.search(rf"\b{bp}\b", text):
            explicit_token = f"{bp} pain"
            explicit.append((explicit_token, 0.99, bp))

    clean = re.sub(r"[^a-zA-Z\s]", " ", text)
    emb = embedding_model.encode([clean])[0]
    sims = cosine_similarity([emb], CAN_EMB)[0]
    idxs = np.argsort(sims)[::-1]
    sem_matches = []
    for i in idxs[:top_n]:
        score = float(sims[i])
        if score >= min_sim:
            sem_matches.append((CANONICAL_SYMPTOMS[i], round(score, 3), CANONICAL_SYMPTOMS[i]))

    merged = []
    seen = set()
    for token, score, src in explicit + sem_matches:
        if token not in seen:
            merged.append((token, score, src))
            seen.add(token)
    return merged[:top_n]

# ============================
# LLM REASONING + FALLBACK
# ============================
def call_reasoning_llm(user_text, extracted_symptoms):
    model = None
    if genai:
        try:
            model = genai.GenerativeModel("gemini-2.5-pro")
        except Exception:
            model = None

    symptom_tokens = [s for s,_,_ in extracted_symptoms]
    prompt = f"""
You are a careful, responsible medical reasoning assistant. Do NOT give a diagnosis. Provide hypotheses.
User raw text: \"{user_text}\"
Extracted symptom tokens: {symptom_tokens}

Task:
1) Return 3 plausible conditions or hypotheses (from most to least likely).
2) For each condition return a JSON object with keys:
   - "condition": short label
   - "confidence": float between 0 and 1
   - "supporting_symptoms": list of the symptom tokens (use tokens exactly)
   - "explanation": 3-5 sentences, clearly linking the supporting symptoms to the condition
   - "care": 1-2 concise home-care suggestions
   - "doctor_advice": when to seek medical help and which specialist

Return EXACTLY a JSON array (a Python list) of these objects.
"""

    if model:
        try:
            resp = model.generate_content(prompt)
            out_text = resp.text.strip()
            parsed = extract_json_list_from_text(out_text)
            if parsed and isinstance(parsed, list) and parsed:
                sanitized = []
                for item in parsed[:5]:
                    try:
                        sanitized.append({
                            "condition": str(item.get("condition", "Unknown")).strip(),
                            "confidence": float(item.get("confidence", 0.5)),
                            "supporting_symptoms": list(item.get("supporting_symptoms", [])),
                            "explanation": str(item.get("explanation", "")).strip(),
                            "care": str(item.get("care", "")).strip(),
                            "doctor_advice": str(item.get("doctor_advice", "")).strip()
                        })
                    except Exception:
                        continue
                if sanitized:
                    return sanitized
        except Exception:
            pass

    return heuristic_reasoning_fallback(user_text, extracted_symptoms)

def heuristic_reasoning_fallback(user_text, extracted_symptoms):
    tokens = [s for s,_,_ in extracted_symptoms]
    preds = []
    used = set()

    for t in tokens:
        if any(x in t for x in ["elbow", "knee", "wrist", "shoulder", "ankle", "joint", "back", "neck"]):
            preds.append({
                "condition": "Local musculoskeletal injury or tendinopathy",
                "confidence": 0.78,
                "supporting_symptoms": [t],
                "explanation": (
                    f"{t.capitalize()} often reflects local strain, tendon irritation, or minor sprain. "
                    "Localized pain, worsened by movement, supports a musculoskeletal cause."
                ),
                "care": "Rest the joint, avoid painful movements, apply ice for 10–15 min several times a day, consider OTC analgesic.",
                "doctor_advice": "See an orthopedic specialist if swelling, severe pain, reduced range of motion, or symptoms persist >72 hours."
            })
            used.add(t)

    for t in tokens:
        if any(x in t for x in ["stomach", "nausea", "vomit", "diarrhea", "loss of appetite"]):
            preds.append({
                "condition": "Gastritis / Dyspepsia or Gastroenteritis",
                "confidence": 0.72,
                "supporting_symptoms": [t],
                "explanation": (
                    f"{t.capitalize()} commonly occurs with stomach inflammation, acid-related irritation, or transient infection. "
                    "Associated symptoms and dietary history influence the exact cause."
                ),
                "care": "Avoid spicy food, favor bland meals, hydrate, consider antacids if appropriate.",
                "doctor_advice": "See a gastroenterologist or GP if severe abdominal pain, persistent vomiting, or blood in stools."
            })
            used.add(t)

    remaining = [t for t in tokens if t not in used]
    if remaining:
        preds.append({
            "condition": "Systemic viral or inflammatory syndrome",
            "confidence": 0.55,
            "supporting_symptoms": remaining,
            "explanation": (
                "When multiple non-specific symptoms (pain, mild fever, fatigue) occur together, a viral or systemic inflammatory cause is possible. "
                "Usually self-limiting but monitoring is important."
            ),
            "care": "Hydrate, rest, and symptomatic relief (paracetamol if needed).",
            "doctor_advice": "If symptoms worsen, develop high fever, or continue beyond 48–72 hours, consult a GP."
        })

    if not preds:
        preds = [{
            "condition": "Non-specific complaint - needs clinical assessment",
            "confidence": 0.50,
            "supporting_symptoms": tokens,
            "explanation": "Symptoms do not clearly match a single system; clinical exam could clarify.",
            "care": "Rest, hydrate, monitor symptoms.",
            "doctor_advice": "Visit General Physician for triage."
        }]

    return preds

# ============================
# SPECIALIST CHOOSER
# ============================
def choose_specialists(extracted_symptoms, llm_predictions):
    specialists = []
    for token, _, _ in extracted_symptoms:
        if token in SPECIALIST_MAP and SPECIALIST_MAP[token] not in specialists:
            specialists.append(SPECIALIST_MAP[token])
    for pred in llm_predictions:
        for s in pred.get("supporting_symptoms", []):
            if s in SPECIALIST_MAP and SPECIALIST_MAP[s] not in specialists:
                specialists.append(SPECIALIST_MAP[s])

    if not specialists:
        cond_text = " ".join([p.get("condition","") for p in llm_predictions]).lower()
        if "stomach" in cond_text or "gastri" in cond_text:
            specialists.append("Gastroenterologist")
        elif "headache" in cond_text or "migraine" in cond_text:
            specialists.append("Neurologist")
        elif "chest" in cond_text or "cardiac" in cond_text:
            specialists.append("Cardiologist")

    if "General Physician" not in specialists:
        specialists.insert(0, "General Physician")
    return specialists

# ============================
# DOCTOR FINDER
# ============================
def find_doctors_nearby(location, specialty="General Physician", radius_m=4000, max_results=6):
    # geocode using Nominatim
    try:
        georesp = requests.get("https://nominatim.openstreetmap.org/search",
                               params={"q": location, "format": "json", "limit": 1},
                               headers={"User-Agent": "AgenticAI"} , timeout=15)
        if georesp.status_code != 200:
            raise Exception("Geocode failed")
        geo = georesp.json()
        if not geo:
            raise Exception("Location not found")
        lat, lon = geo[0]["lat"], geo[0]["lon"]
    except Exception:
        gmap_link = f"https://www.google.com/maps/search/{specialty}+near+{location.replace(' ', '+')}"
        return f"🗺️ Unable to geolocate {location}. Try Google Maps: {gmap_link}"

    overpass_q = f"""
[out:json][timeout:25];
(
  node["amenity"="hospital"](around:{radius_m},{lat},{lon});
  node["amenity"="clinic"](around:{radius_m},{lat},{lon});
  node["healthcare"="doctor"](around:{radius_m},{lat},{lon});
  way["amenity"="hospital"](around:{radius_m},{lat},{lon});
  way["amenity"="clinic"](around:{radius_m},{lat},{lon});
);
out center;
"""
    try:
        resp = requests.get("https://overpass-api.de/api/interpreter", params={"data": overpass_q}, timeout=30, headers={"User-Agent":"AgenticAI"})
        if resp.status_code != 200 or not resp.text.strip():
            raise Exception("Overpass empty")
        data = resp.json().get("elements", [])
        if not data:
            raise Exception("No elements")
        specialty_lower = specialty.lower()
        filtered = []
        for el in data:
            tags = el.get("tags", {})
            name = tags.get("name", "")
            speciality_tag = tags.get("speciality") or tags.get("specialty") or ""
            name_and_tags = " ".join([name, speciality_tag]).lower()
            if specialty_lower in name_and_tags:
                filtered.append((el, 1))
            elif name:
                filtered.append((el, 0.6))
            else:
                filtered.append((el, 0.2))

        filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)
        out_lines = []
        seen = set()
        for el, score in filtered_sorted[:max_results*2]:
            tags = el.get("tags", {})
            name = tags.get("name", "Unnamed Facility")
            if name in seen:
                continue
            seen.add(name)
            lat_e = el.get("lat") or el.get("center", {}).get("lat")
            lon_e = el.get("lon") or el.get("center", {}).get("lon")
            if lat_e and lon_e:
                map_link = f"https://www.google.com/maps?q={lat_e},{lon_e}"
            else:
                map_link = f"https://www.google.com/maps/search/{name.replace(' ', '+')}"
            contact = tags.get("phone") or tags.get("contact:phone") or ""
            extra = f" — {contact}" if contact else ""
            out_lines.append(f"🏥 **{name}**{extra} — [View on Map]({map_link})")
            if len(out_lines) >= max_results:
                break
        if out_lines:
            return "\n\n".join(out_lines)
        else:
            raise Exception("No plausible results")
    except Exception:
        gmap_link = f"https://www.google.com/maps/search/{specialty}+near+{location.replace(' ', '+')}"
        return f"🗺️ Unable to fetch local doctors. Try Google Maps: {gmap_link}"

# ============================
# EXPLAINABLE VISUALS
# ============================
def show_explainability(predictions, extracted_symptoms):
    if not predictions:
        st.warning("No predictions to explain.")
        return

    df = pd.DataFrame(predictions)
    fig = px.bar(df, x="condition", y="confidence", text="confidence",
                 color="confidence", color_continuous_scale="Viridis",
                 title="Model Confidence per Condition")
    st.plotly_chart(fig, use_container_width=True)

    mild_keywords = ["cold", "flu", "fatigue", "stress", "viral"]
    moderate_keywords = ["infection", "gastritis", "pneumonia", "migraine", "tendinopathy"]
    severe_keywords = ["stroke", "heart attack", "cancer", "sepsis"]
    mild = sum([p["confidence"] for p in predictions if any(k in p["condition"].lower() for k in mild_keywords)])
    moderate = sum([p["confidence"] for p in predictions if any(k in p["condition"].lower() for k in moderate_keywords)])
    severe = sum([p["confidence"] for p in predictions if any(k in p["condition"].lower() for k in severe_keywords)])
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=[mild, moderate, severe, mild],
                                    theta=["Mild", "Moderate", "Severe", "Mild"],
                                    fill="toself",
                                    name="Severity"))
    radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 2])), showlegend=False,
                        title="Severity Distribution (heuristic)")
    st.plotly_chart(radar, use_container_width=True)

    symptoms = [s for s,_,_ in extracted_symptoms]
    if not symptoms:
        st.info("No canonical symptoms detected for matrix.")
    else:
        matrix = []
        for pred in predictions:
            row = []
            supp = [x.lower() for x in pred.get("supporting_symptoms", [])]
            for s in symptoms:
                if s.lower() in supp:
                    row.append(1.0)
                else:
                    row.append(0.1 if s.lower() in pred.get("condition", "").lower() else 0.0)
            matrix.append(row)
        if any(sum(r) > 0 for r in matrix):
            fig2 = px.imshow(np.array(matrix),
                             labels=dict(x="Symptom", y="Condition", color="Support"),
                             x=symptoms, y=[p["condition"] for p in predictions],
                             color_continuous_scale="Oranges")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Heatmap not informative for these inputs.")

    st.subheader("Detailed Explanation (descriptive)")
    for pred in predictions:
        st.markdown(f"**{pred['condition']}**  —  Confidence: **{pred['confidence']:.2f}**")
        st.markdown(f"> **Supporting symptoms:** {', '.join(pred.get('supporting_symptoms', []) ) or 'None listed.'}")
        st.markdown(f"> **Explanation:** {pred.get('explanation', '')}")
        st.markdown(f"> **Self-care:** {pred.get('care', '')}")
        st.markdown(f"> **When to see a doctor / specialist advice:** {pred.get('doctor_advice', '')}")
        st.markdown("---")

# ============================
# MEMORY
# ============================
MEM_FN = "memory1.json"

def save_consultation(record):
    data = []
    if os.path.exists(MEM_FN):
        try:
            with open(MEM_FN, "r") as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(record)
    with open(MEM_FN, "w") as f:
        json.dump(data, f, indent=2)

def load_consultations():
    if not os.path.exists(MEM_FN):
        return []
    try:
        with open(MEM_FN, "r") as f:
            return json.load(f)
    except Exception:
        return []

def delete_consultation_at_index(idx):
    data = load_consultations()
    if 0 <= idx < len(data):
        data.pop(idx)
        with open(MEM_FN, "w") as f:
            json.dump(data, f, indent=2)
        return True
    return False

# ============================
# STREAMLIT UI
# ============================
st.title("🧬 Agentic AI Healthcare Assistant — Generic & Explainable")
st.caption("Descriptive explanations, multi-system reasoning, and actionable local help.")

col_main, col_info = st.columns([3,1])
with col_info:
    st.info("Pro tip: be as specific as possible (e.g., 'pain in my right elbow and burning stomach pain').")

location = st.sidebar.text_input("Enter city (for nearby doctors):", "Mumbai, India")

# ----------------------------
# Past consultations in sidebar
# ----------------------------
if st.sidebar.button("View Past Consultations"):
    hist = load_consultations()
    if not hist:
        st.sidebar.info("No past consultations.")
    else:
        # iterate in reverse (most recent first)
        for i, rec in enumerate(hist[::-1], 1):
            # compute original index in list (for deletion)
            orig_idx = len(hist) - i
            title = rec.get("predictions", [{}])[0].get("condition") if rec.get("predictions") else rec.get("condition", "-")
            conf = rec.get("predictions", [{}])[0].get("confidence") if rec.get("predictions") else rec.get("confidence", "")
            with st.sidebar.expander(f"{i}. {title} ({conf})"):
                st.markdown(f"**Query:** {rec.get('query','-')}")
                st.markdown(f"**When:** {rec.get('timestamp','-')}")
                st.markdown(f"**Detected symptoms:** {', '.join(rec.get('symptoms', [])) or '-'}")

                # Show saved predictions (detailed) inside expander
                if rec.get("predictions"):
                    st.markdown("**Predictions:**")
                    for p in rec.get("predictions", []):
                        st.markdown(f"- **{p.get('condition','-')}**  — Confidence: **{p.get('confidence',0):.2f}**")
                        st.markdown(f"  - Supporting symptoms: {', '.join(p.get('supporting_symptoms', [])) or '-'}")
                        st.markdown(f"  - Explanation: {p.get('explanation','-')}")
                        st.markdown(f"  - Self-care: {p.get('care','-')}")
                        st.markdown(f"  - Doctor advice: {p.get('doctor_advice','-')}")
                else:
                    st.markdown("_No predictions saved for this record._")

                # Specialists
                st.markdown("**Recommended specialists:**")
                for s in rec.get("specialists", ["General Physician"]):
                    st.write("- " + s)

                # Buttons to show nearby doctors for this consultation or delete the record
                cols = st.columns([1,1])
                if cols[0].button(f"Show nearby doctors #{i}", key=f"near_{i}"):
                    # display in the expander below buttons
                    for spec in rec.get("specialists", [])[:3]:
                        st.markdown(f"**{spec} near {rec.get('location', location)}:**")
                        res = find_doctors_nearby(rec.get("location", location), spec)
                        st.markdown(res)

                if cols[1].button(f"Delete #{i}", key=f"del_{i}"):
                    # delete by original index
                    deleted = delete_consultation_at_index(orig_idx)
                    if deleted:
                        st.sidebar.success(f"Deleted consultation #{i}. Refresh to see changes.")
                    else:
                        st.sidebar.error("Could not delete the consultation.")

st.markdown("---")

# ----------------------------
# Main chat input and handling
# ----------------------------
user_input = st.chat_input("Describe your symptoms (e.g. 'I have pain in my elbow and stomach')")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    # 1) extract symptoms
    with st.spinner("🔍 Extracting symptoms..."):
        extracted = extract_symptoms_with_parts(user_input)
        if not extracted:
            st.warning("No clear symptoms detected. Try rephrasing or adding body parts (e.g., 'right elbow pain').")
        else:
            st.success("Detected: " + ", ".join([f"{s} ({score})" for s,score,_ in extracted]))

    # 2) LLM reasoning + fallback
    with st.spinner("🧠 Reasoning (LLM + fallback)..."):
        predictions = call_reasoning_llm(user_input, extracted)
        for p in predictions:
            p["confidence"] = max(0.0, min(1.0, float(p.get("confidence", 0.5))))
            p["supporting_symptoms"] = [str(x) for x in p.get("supporting_symptoms", [])]

    if not predictions:
        predictions = heuristic_reasoning_fallback(user_input, extracted)

    # 3) Present top prediction
    top = predictions[0]
    st.markdown(f"### 🩺 Most Probable: **{top['condition']}**  — Confidence: **{top['confidence']:.2f}**")
    st.markdown(f"**Explanation:** {top.get('explanation','No explanation available.')}")
    st.info(f"**Self-care:** {top.get('care','No care advice available.')}")
    st.warning(f"**When to consult a doctor:** {top.get('doctor_advice','If concerned, see a GP.')}")

    # 4) Explainability visuals
    with st.expander("📊 Explainable AI details (confidence, severity, symptom→condition support)"):
        show_explainability(predictions, extracted)

    # 5) Choose specialists and list
    specialists = choose_specialists(extracted, predictions)
    st.subheader("👩‍⚕️ Recommended specialists (triage order)")
    for spec in specialists:
        st.markdown(f"- **{spec}**")

    # 6) Show nearby for top 2 specialists inline in main panel
    st.subheader("🏥 Nearby doctors / clinics (quick view)")
    for spec in specialists[:2]:
        st.markdown(f"**{spec} near {location}:**")
        res = find_doctors_nearby(location, spec)
        st.markdown(res)

    # 7) Save full consultation (including predictions + specialists)
    record = {
        "query": user_input,
        "symptoms": [s for s,_,_ in extracted],
        "predictions": predictions,
        "specialists": specialists,
        "location": location,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    save_consultation(record)
    st.success("Consultation saved to local history (memory.json).")
