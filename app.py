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
import google.generativeai as genai

# ============================
# CONFIG
# ============================
load_dotenv()
GENIE_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GENIE_API_KEY")
if GENIE_KEY:
    genai.configure(api_key=GENIE_KEY)

st.set_page_config(page_title="Agentic AI Healthcare Assistant (v3)", layout="wide", page_icon="🧬")

# ============================
# HELPERS / CACHING
# ============================
@st.cache_resource
def load_embedding_model():
    # lightweight model; change if you prefer a bigger one
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# Small canonical symptom list for semantic matching (extendable)
CANONICAL_SYMPTOMS = [
    "headache", "fever", "cough", "sore throat", "runny nose", "shortness of breath",
    "chest pain", "stomach ache", "nausea", "vomiting", "diarrhea",
    "back pain", "neck pain", "elbow pain", "knee pain", "wrist pain",
    "joint pain", "muscle pain", "fatigue", "dizziness", "rash",
    "itching", "anxiety", "depression", "weight loss", "loss of appetite",
    "vision changes", "weakness", "numbness"
]

CAN_EMB = embedding_model.encode(CANONICAL_SYMPTOMS, show_progress_bar=False)

# Specialist mapping (expandable)
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
    # fallback
    "default": "General Physician"
}

# Utility: robust JSON parse helper
def extract_json_list_from_text(text):
    """
    Finds the first JSON list in a text blob and returns parsed Python list.
    """
    import re, json
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

# ============================
# SYMPTOM EXTRACTION (granular)
# ============================
def extract_symptoms_with_parts(user_text, min_sim=0.52, top_n=6):
    """
    Returns a list of (symptom_token, score, source_phrase).
    - Detects explicit body parts (elbow, knee etc.) -> high confidence tokens like 'elbow pain'
    - Uses sentence-transformer semantic similarity to canonical symptoms
    """
    text = user_text.lower()
    # 1) explicit body part detection (keeps granularity)
    body_parts = ["elbow", "knee", "shoulder", "wrist", "ankle", "back", "neck", "hip", "jaw"]
    explicit = []
    for bp in body_parts:
        if re.search(rf"\b{bp}\b", text):
            # attempt to detect if user said "pain in elbow" or "elbow pain"
            # create token "<bp> pain"
            explicit_token = f"{bp} pain"
            explicit.append((explicit_token, 0.99, bp))

    # 2) semantic detection against canonical list
    clean = re.sub(r"[^a-zA-Z\s]", " ", text)
    emb = embedding_model.encode([clean])[0]
    sims = cosine_similarity([emb], CAN_EMB)[0]
    idxs = np.argsort(sims)[::-1]
    sem_matches = []
    for i in idxs[:top_n]:
        score = float(sims[i])
        if score >= min_sim:
            sem_matches.append((CANONICAL_SYMPTOMS[i], round(score, 3), CANONICAL_SYMPTOMS[i]))
    # merge while avoiding duplicates
    merged = []
    seen = set()
    for token, score, src in explicit + sem_matches:
        if token not in seen:
            merged.append((token, score, src))
            seen.add(token)
    return merged[:top_n]

# ============================
# LLM REASONING (structured + robust fallback)
# ============================
def call_reasoning_llm(user_text, extracted_symptoms):
    """
    Ask Gemini (or the configured model) to reason and produce a JSON list of conditions with
    supporting_symptoms, explanation, care, doctor_advice and confidence.
    If model output is missing/invalid, fallback to heuristic generation.
    """
    # ensure API configured
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
    except Exception:
        model = None

    # LLM prompt: require supporting_symptoms (use extracted symptom tokens)
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

Return EXACTLY a JSON array (a Python list) of these objects. Example:
[
  {{
    "condition": "Migraine",
    "confidence": 0.82,
    "supporting_symptoms": ["headache","nausea"],
    "explanation": "Migraine commonly presents with severe unilateral headache ...",
    "care": "rest in a dark room, hydration, OTC analgesics.",
    "doctor_advice": "See a neurologist if headaches are new, severe, or with visual disturbances."
  }},
  ...
]
"""
    # call LLM and robustly parse
    if model:
        try:
            resp = model.generate_content(prompt)
            out_text = resp.text.strip()
            parsed = extract_json_list_from_text(out_text)
            if parsed and isinstance(parsed, list) and parsed:
                # ensure fields exist; sanitize
                sanitized = []
                for item in parsed[:5]:
                    sanitized.append({
                        "condition": str(item.get("condition", "Unknown")).strip(),
                        "confidence": float(item.get("confidence", 0.5)),
                        "supporting_symptoms": list(item.get("supporting_symptoms", [])),
                        "explanation": str(item.get("explanation", "")).strip(),
                        "care": str(item.get("care", "")).strip(),
                        "doctor_advice": str(item.get("doctor_advice", "")).strip()
                    })
                return sanitized
        except Exception:
            # fall through to fallback
            pass

    # FALLBACK: heuristic generator that splits into per-system hypotheses
    return heuristic_reasoning_fallback(user_text, extracted_symptoms)

def heuristic_reasoning_fallback(user_text, extracted_symptoms):
    """
    Create 2-3 plausible hypotheses heuristically, grounded in extracted symptoms.
    This is used when the LLM does not produce usable JSON.
    """
    tokens = [s for s,_,_ in extracted_symptoms]
    preds = []

    # If multiple clear body parts or distinct systems -> produce separate hypotheses
    # Example: elbow pain -> local musculoskeletal; stomach ache -> GI
    used = set()

    # handle joints / body-parts -> Orthopedic
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

    # handle GI symptoms
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

    # if nothing matched above, create a general systemic hypothesis
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

    # Ensure we have at least one hypothesis
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
    """
    Return a list of recommended specialists - possibly multiple - ordered by priority.
    Logic:
     - First, look for body-part tokens in extracted symptoms mapped in SPECIALIST_MAP.
     - Then, inspect llm_predictions supporting_symptoms for mapped tokens.
     - If multiple systems implicated, return multiple specialists (GP + specialists)
    """
    specialists = []
    # check explicit extracted symptoms first
    for token, _, _ in extracted_symptoms:
        if token in SPECIALIST_MAP and SPECIALIST_MAP[token] not in specialists:
            specialists.append(SPECIALIST_MAP[token])
    # check supporting symptoms inside model preds
    for pred in llm_predictions:
        for s in pred.get("supporting_symptoms", []):
            if s in SPECIALIST_MAP and SPECIALIST_MAP[s] not in specialists:
                specialists.append(SPECIALIST_MAP[s])
    # If none, look at predicted condition text for hints
    if not specialists:
        cond_text = " ".join([p.get("condition","") for p in llm_predictions]).lower()
        if "stomach" in cond_text or "gastri" in cond_text:
            specialists.append("Gastroenterologist")
        elif "headache" in cond_text or "migraine" in cond_text:
            specialists.append("Neurologist")
        elif "chest" in cond_text or "cardiac" in cond_text:
            specialists.append("Cardiologist")
    # Always include GP as first-line triage option
    if "General Physician" not in specialists:
        specialists.insert(0, "General Physician")
    return specialists

# ============================
# DOCTOR FINDER (filtered + fallback)
# ============================
def find_doctors_nearby(location, specialty="General Physician", radius_m=4000, max_results=6):
    """
    Uses Nominatim to geocode location and Overpass to find nearby healthcare nodes.
    Filters results to prefer ones that mention the specialty; otherwise returns named clinics.
    Falls back to Google Maps search link on failure.
    """
    # geocode
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

    # build Overpass query (hospitals, clinics, doctors)
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
        # filter by specialty mention in tags or name
        specialty_lower = specialty.lower()
        filtered = []
        for el in data:
            tags = el.get("tags", {})
            name = tags.get("name", "")
            # check explicit speciality tag
            speciality_tag = tags.get("speciality") or tags.get("specialty") or ""
            name_and_tags = " ".join([name, speciality_tag]).lower()
            if specialty_lower in name_and_tags:
                filtered.append((el, 1))
            elif name:  # prefer named facilities
                filtered.append((el, 0.6))
            else:
                filtered.append((el, 0.2))
        # sort by score and uniqueness
        filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)
        out_lines = []
        seen = set()
        for el, score in filtered_sorted[:max_results*2]:
            tags = el.get("tags", {})
            name = tags.get("name", "Unnamed Facility")
            # dedupe by name
            if name in seen:
                continue
            seen.add(name)
            # get coords
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
    """
    - Bar of confidences
    - Severity radar
    - Symptom->Condition heatmap (support matrix)
    - Clear textual per-condition explanations (descriptive)
    """
    if not predictions:
        st.warning("No predictions to explain.")
        return

    df = pd.DataFrame(predictions)
    # Confidence bar
    fig = px.bar(df, x="condition", y="confidence", text="confidence",
                 color="confidence", color_continuous_scale="Viridis",
                 title="Model Confidence per Condition")
    st.plotly_chart(fig, use_container_width=True)

    # Severity radar - derive severity tokens heuristically from condition label
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

    # Symptom->Condition support matrix (heatmap)
    symptoms = [s for s,_,_ in extracted_symptoms]
    if not symptoms:
        st.info("No canonical symptoms detected for matrix.")
    else:
        matrix = []
        for pred in predictions:
            row = []
            supp = [x.lower() for x in pred.get("supporting_symptoms", [])]
            for s in symptoms:
                # if the model listed the symptom as supporting -> 1.0
                if s.lower() in supp:
                    row.append(1.0)
                else:
                    # weak heuristic: if token word appears in condition label, small weight
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

    # Textual descriptive explainability
    st.subheader("Detailed Explanation (descriptive)")
    for pred in predictions:
        st.markdown(f"**{pred['condition']}**  —  Confidence: **{pred['confidence']:.2f}**")
        st.markdown(f"> **Supporting symptoms:** {', '.join(pred.get('supporting_symptoms', []) ) or 'None listed.'}")
        st.markdown(f"> **Explanation:** {pred.get('explanation', '')}")
        st.markdown(f"> **Self-care:** {pred.get('care', '')}")
        st.markdown(f"> **When to see a doctor / specialist advice:** {pred.get('doctor_advice', '')}")
        st.markdown("---")

# ============================
# MEMORY (simple)
# ============================
def save_consultation(record):
    fn = "memory.json"
    data = []
    if os.path.exists(fn):
        try:
            with open(fn, "r") as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(record)
    with open(fn, "w") as f:
        json.dump(data, f, indent=2)

def load_consultations():
    fn = "memory.json"
    if not os.path.exists(fn):
        return []
    try:
        with open(fn, "r") as f:
            return json.load(f)
    except Exception:
        return []

# ============================
# STREAMLIT UI
# ============================
st.title("🧬 Agentic AI Healthcare Assistant — Generic & Explainable")
st.caption("Descriptive explanations, multi-system reasoning, and actionable local help.")

col1, col2 = st.columns([3,1])
with col2:
    st.info("Pro tip: be as specific as possible (e.g., 'pain in my right elbow and burning stomach pain').")

location = st.sidebar.text_input("Enter city (for nearby doctors):", "Mumbai, India")
if st.sidebar.button("View Past Consultations"):
    hist = load_consultations()
    if not hist:
        st.sidebar.info("No past consultations.")
    else:
        for i, rec in enumerate(hist[::-1], 1):
            with st.sidebar.expander(f"{i}. {rec.get('condition','-')} ({rec.get('confidence', '')})"):
                st.markdown(f"**Query:** {rec.get('query')}")
                st.markdown(f"**Symptoms:** {', '.join(rec.get('symptoms',[]))}")
                st.caption(f"Location: {rec.get('location','-')}")

user_input = st.chat_input("Describe your symptoms (e.g. 'I have pain in my elbow and stomach')")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    # Step 1: extract symptoms (granular)
    with st.spinner("🔍 Extracting symptoms..."):
        extracted = extract_symptoms_with_parts(user_input)
        if not extracted:
            st.warning("No clear symptoms detected. Try rephrasing or adding body parts (e.g., 'right elbow pain').")
        else:
            st.success("Detected: " + ", ".join([f"{s} ({score})" for s,score,_ in extracted]))

    # Step 2: LLM reasoning (structured) with fallback
    with st.spinner("🧠 Reasoning (LLM + fallback)..."):
        predictions = call_reasoning_llm(user_input, extracted)
        # normalize confidence to 0-1 and ensure fields
        for p in predictions:
            p["confidence"] = max(0.0, min(1.0, float(p.get("confidence", 0.5))))
            p["supporting_symptoms"] = [str(x) for x in p.get("supporting_symptoms", [])]

    # If predictions are empty, use fallback heuristics
    if not predictions:
        predictions = heuristic_reasoning_fallback(user_input, extracted)

    # Step 3: Present top prediction and descriptive explanation
    top = predictions[0]
    st.markdown(f"### 🩺 Most Probable: **{top['condition']}**  — Confidence: **{top['confidence']:.2f}**")
    st.markdown(f"**Explanation:** {top.get('explanation','No explanation available.')}")
    st.info(f"**Self-care:** {top.get('care','No care advice available.')}")
    st.warning(f"**When to consult a doctor:** {top.get('doctor_advice','If concerned, see a GP.')}")

    # Step 4: Show explainability visuals (bar, radar, heatmap, textual)
    with st.expander("📊 Explainable AI details (confidence, severity, symptom→condition support)"):
        show_explainability(predictions, extracted)

    # Step 5: Choose specialists (could be multiple) and show nearby doctors for each
    specialists = choose_specialists(extracted, predictions)
    st.subheader("👩‍⚕️ Recommended specialists (triage order)")
    for spec in specialists:
        st.markdown(f"- **{spec}**")

    # For each non-GP specialist (or top 2 specialists), fetch nearby doctors/hospitals
    st.subheader("🏥 Nearby doctors / clinics")
    for spec in specialists[:3]:
        st.markdown(f"**{spec} near {location}:**")
        res = find_doctors_nearby(location, spec)
        st.markdown(res)

    # Save consultation to memory
    save_consultation({
        "query": user_input,
        "symptoms": [s for s,_,_ in extracted],
        "condition": top["condition"],
        "confidence": top["confidence"],
        "location": location
    })

    st.success("Consultation saved to local history (memory.json).")
