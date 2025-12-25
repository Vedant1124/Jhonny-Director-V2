import os
import json
import re
import requests
from typing import Annotated, List, TypedDict, Optional
from dotenv import load_dotenv

# --- LANGGRAPH ---
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

# --- AI MODELS ---
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

load_dotenv()

# =========================
# 1. CONFIG & MODELS
# =========================
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vision Fallback
try:
    blip_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=-1)
except:
    blip_pipe = None

HF_TOKEN = os.getenv("HF_TOKEN")

# PATHS
BASE = r"D:\Data for Jhonny Director\Code for new data\data"
PATH_ENGINE = os.path.join(BASE, "Engine")
PATH_ORCH = os.path.join(BASE, "Orchestrator")
PATH_IMG_FAISS = os.path.join(BASE, "images extracted")

# =========================
# 2. STATE DEFINITION (FULL SCHEMA)
# =========================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    image_path: Optional[str]
    image_analysis: dict
    status: str             # CLARIFYING / READY
    
    # --- A. USER QUESTION SYSTEM (INPUTS) ---
    # We track every single variable required by your document
    selections: dict  
    # Keys:
    # project_type, output_depth
    # shot_count, progression
    # shot_family, motion, lighting, environment, time_of_day, vfx
    # emotion, intensity
    # lens, sensor
    # camera_brand, film_stock
    # platform

    missing_steps: List[str] # Ordered list of steps to complete
    current_step: str        # The specific step we are asking about now
    
    sequence: dict
    qa: dict
    final_output: str

# =========================
# 3. UTILS
# =========================
def extract_json(text: str) -> dict:
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except: return {}

def rag(folder, index_name, query, k=3):
    try:
        store = FAISS.load_local(folder, embeddings, index_name=index_name, allow_dangerous_deserialization=True)
        docs = store.similarity_search(query, k=k)
        return "\n".join(d.page_content for d in docs)
    except: return ""

def hf_api_caption(path):
    if not HF_TOKEN: return None
    api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    with open(path, "rb") as f: data = f.read()
    response = requests.post(api_url, headers={"Authorization": f"Bearer {HF_TOKEN}"}, data=data)
    try: return response.json()[0]['generated_text']
    except: return None

# =========================
# 4. NODES
# =========================

def image_ingestion_node(state: AgentState):
    """Analyzes image for context."""
    path = state.get("image_path")
    caption = ""
    if path and os.path.exists(path):
        if blip_pipe:
            try: caption = blip_pipe(path)[0]['generated_text']
            except: pass
        if not caption:
            caption = hf_api_caption(path)
        if not caption:
            caption = "Image uploaded but analysis failed."
    else:
        caption = "No visual reference provided yet."

    rag_ctx = rag(PATH_IMG_FAISS, "image_master", caption) if "No visual" not in caption else ""
    state["image_analysis"] = {"caption": caption, "rag_context": rag_ctx}
    return state

def supervisor_node(state: AgentState):
    """
    THE ANALYST.
    Strictly follows your A->I Order.
    Identifies what has been answered and calculates the next logical step.
    """
    user_text = state["messages"][-1].content
    selections = state.get("selections", {})
    
    # 1. DEFINING THE STRICT ORDER OF OPERATIONS
    # These map to your document sections (A, B, C, D, E, F, G)
    # We group small related items to prevent 20-turn fatigue, but strictly follow the flow.
    steps_order = [
        "project_context",    # A1: Type (Ad/Film) + Depth
        "logic_strategy",     # A2: Shot Count + Progression
        "cinematic_pillars",  # B: Family + Motion + Lighting
        "environment",        # B4: Location + Time
        "vfx",                # B5: VFX
        "emotion_story",      # C: Emotion + Intensity
        "optics",             # D/E: Lens + Sensor
        "camera_look",        # F: Brand + Stock
        "delivery"            # G: Platform
    ]

    # RAG for context
    rules = rag(PATH_ENGINE, "indexG", user_text)

    prompt = f"""
    ### ROLE: SYSTEM ANALYST
    Update the technical brief based on the user's latest message.
    
    USER MESSAGE: "{user_text}"
    CURRENT SELECTIONS: {json.dumps(selections)}
    
    ### TASK:
    1. **Extract**: Look for keywords matching the schema below. Update `updated_selections`.
       - *Context*: Short film, Ad, Brand video...
       - *Logic*: 1 shot, 5 shots, Establish->Reveal...
       - *Pillars*: EWS, MCU, Dolly, Soft light, Neon...
       - *Environment*: Interior, Forest, Morning, Night...
       - *VFX*: Subtle, Heavy, Fire...
       - *Emotion*: Happy, Tense, Fear...
       - *Optics*: 50mm, Macro, Anamorphic, Full Frame...
       - *Look*: ARRI, Kodak, Log...
       - *Platform*: YouTube, Netflix, Instagram...
    
    2. **Auto-Fill**: If user says "Let Jhonny Decide" (or similar), fill ALL remaining null fields with "Auto".

    Return JSON ONLY:
    {{
        "updated_selections": {{...}}
    }}
    """
    res = llm.invoke([SystemMessage(content=prompt)] + state["messages"])
    data = extract_json(res.content)
    
    # Update State
    new_sels = data.get("updated_selections", {})
    updated_selections = {**selections, **new_sels}
    
    # Calculate Missing Steps
    missing_steps = []
    
    # Check A1: Context
    if not (updated_selections.get("project_type") and updated_selections.get("output_depth")):
        missing_steps.append("project_context")
    
    # Check A2: Logic
    if not (updated_selections.get("shot_count") and updated_selections.get("progression")):
        missing_steps.append("logic_strategy")

    # Check B: Pillars (Visuals)
    if not (updated_selections.get("shot_family") and updated_selections.get("motion") and updated_selections.get("lighting")):
        missing_steps.append("cinematic_pillars")
    
    # Check B4: Environment
    if not (updated_selections.get("environment") and updated_selections.get("time_of_day")):
        missing_steps.append("environment")

    # Check B5: VFX
    if not updated_selections.get("vfx"):
        missing_steps.append("vfx")

    # Check C: Emotion
    if not (updated_selections.get("emotion") and updated_selections.get("intensity")):
        missing_steps.append("emotion_story")
        
    # Check D/E: Optics
    if not (updated_selections.get("lens") and updated_selections.get("sensor")):
        missing_steps.append("optics")

    # Check F: Look
    if not (updated_selections.get("camera_brand") and updated_selections.get("film_stock")):
        missing_steps.append("camera_look")

    # Check G: Delivery
    if not updated_selections.get("platform"):
        missing_steps.append("delivery")

    state["selections"] = updated_selections
    state["missing_steps"] = missing_steps
    state["current_step"] = missing_steps[0] if missing_steps else "complete"
    state["status"] = "READY" if not missing_steps else "CLARIFYING"
    
    return state

def questioner_node(state: AgentState):
    """
    THE INTERVIEWER.
    Asks the specific questions defined in your document for the Current Step.
    """
    current_step = state.get("current_step")
    img_data = state.get("image_analysis", {})
    caption = img_data.get("caption", "")
    selections = state.get("selections", {})
    
    prompt = f"""
    ### ROLE: DIRECTOR JHONNY
    You are a Senior Creative Producer. Polite, professional, and efficient.
    
    CONTEXT:
    - Image: {caption}
    - Brief so far: {json.dumps(selections)}
    - **CURRENT STEP**: {current_step}

    ### TASK:
    Ask the user specifically about the missing variables for this step. Use the options below.

    **STEP GUIDES:**
    
    1. **project_context** (A1):
       - Ask: "What are we creating? (Short film, Ad, Brand video, Music video, Social/UGC?)"
       - Ask: "And what Output Depth? (Simple, Detailed, Storyboard-level?)"
    
    2. **logic_strategy** (A2):
       - Ask: "How many shots? (1, 3, 5, or Auto?)"
       - Ask: "Shot Progression? (Establish->Reveal, Calm->Peak, or Auto?)"

    3. **cinematic_pillars** (B1-B3):
       - Ask: "Preferred Shot Family? (Wide, Close-Up, POV, Auto?)"
       - Ask: "Camera Motion? (Static, Dolly, Handheld?)"
       - Ask: "Lighting? (Soft, Hard, Neon, High Key?)"

    4. **environment** (B4):
       - Ask: "Scene Location? (Interior, Studio, Forest, Sci-Fi?)"
       - Ask: "Time of Day? (Morning, Night, Auto?)"

    5. **vfx** (B5):
       - Ask: "VFX Enhancements? (None, Subtle Fog/Glow, Moderate Particles, Heavy Hologram?)"

    6. **emotion_story** (C):
       - Ask: "Core Emotion? (Calm, Happy, Tense, Fear, Power?)"
       - Ask: "Intensity? (Low, Medium, High?)"

    7. **optics** (D/E):
       - Ask: "Lens Choice? (50mm, Wide 24mm, Macro, Anamorphic?)"
       - Ask: "Sensor Size? (Full Frame, APS-C?)"

    8. **camera_look** (F):
       - Ask: "Camera Brand Simulation? (ARRI Soft, RED Punchy, SONY Clean?)"
       - Ask: "Film Stock? (Kodak Vision3, Vintage, Fuji?)"

    9. **delivery** (G):
       - Ask: "Target Platform? (YouTube, Netflix, Instagram?)"

    **ALWAYS**:
    - Remind them: "If you're unsure, just say — **Let Jhonny Decide**."
    - Be concise. Combine the questions for this step into one natural message.
    """
    res = llm.invoke([SystemMessage(content=prompt)] + state["messages"])
    state["final_output"] = res.content
    return state

def orchestrator_node(state: AgentState):
    """
    THE SYSTEM BACKEND.
    This applies the "Systems work behind the scenes" logic (D2-D5, E2, F3, H, I).
    """
    sel = state.get("selections", {})
    caption = state.get("image_analysis", {}).get("caption", "")
    
    # We pass the full selections to the Orchestrator RAG
    master = rag(PATH_ORCH, "merged-multipillar", str(sel))
    
    prompt = f"""
    ### MASTER ORCHESTRATOR
    You are the 'System working behind the scenes'.
    Apply the Logic Modules and Mappings strictly.

    USER SELECTIONS: {json.dumps(sel)}
    VISUAL CONTEXT: {caption}
    MASTER DATA: {master}

    ### EXECUTION PROTOCOL:
    1. **Apply Logic Module 1 (Shot Logic)**: Use '{sel.get('shot_count')}' shots in '{sel.get('progression')}' style.
    2. **Apply D2-D5 (Lens Logic)**: Map '{sel.get('emotion')}' to Lens if Auto. Apply Lens<->Lighting rules.
    3. **Apply E2-E3 (Physics)**: Calculate Depth of Field based on '{sel.get('lens')}' and '{sel.get('sensor')}'.
    4. **Apply F (Color)**: Simulate '{sel.get('camera_brand')}' look.
    5. **Apply G (Delivery)**: Use '{sel.get('platform')}' aspect ratios and codecs.
    6. **Apply H (Audio)**: Map Emotion to Music.

    ### OUTPUT:
    Generate the **Sequence Object** containing:
    - Cinematic Intent
    - The Shot List (Framing, Motion, Lighting, Lens, VFX, Audio Notes)
    - Technical Metadata (Sensor, Export Settings, Color Science)
    
    Return JSON.
    """
    res = llm.invoke([SystemMessage(content=prompt)] + state["messages"])
    state["sequence"] = extract_json(res.content)
    return state

def delivery_node(state: AgentState):
    """
    THE OUTPUT SYSTEM (Section I).
    Generates the 4 required outputs (Text, JSON, Storyboard).
    """
    seq = state.get("sequence", {})
    # Default to text/storyboard combo for clarity, but ideally we ask I1-I3 preference
    # The prompt below generates a comprehensive package.
    
    prompt = f"""
    You are Director Jhonny. Produce the FINAL OUTPUT PACKAGE (Section I).
    
    SEQUENCE DATA: {json.dumps(seq)}

    ### REQUIRED OUTPUTS:
    
    1. **I1. Text Output**: A human-readable cinematic description of the scene flow.
    2. **I2. JSON Output**: The full technical manifest (Lens, Lights, Camera, Audio).
    3. **I3. Multi-Scene Storyboard**: A structured list (Shot 1 -> Shot 2 -> etc.) with Camera/Light/Lens/VFX details.
    
    Format this beautifully using Markdown.
    """
    res = llm.invoke([SystemMessage(content=prompt)] + state["messages"])
    state["final_output"] = res.content
    return state

# =========================
# 5. GRAPH ASSEMBLY
# =========================
def build_director_graph():
    g = StateGraph(AgentState)
    g.add_node("IMAGE", image_ingestion_node)
    g.add_node("SUPERVISOR", supervisor_node)
    g.add_node("QUESTIONER", questioner_node)
    g.add_node("ORCHESTRATOR", orchestrator_node)
    g.add_node("DELIVERY", delivery_node)

    g.add_edge(START, "IMAGE")
    g.add_edge("IMAGE", "SUPERVISOR")
    
    # Conditional Routing based on Missing Steps
    g.add_conditional_edges(
        "SUPERVISOR",
        lambda s: "ORCHESTRATOR" if s["status"] == "READY" else "QUESTIONER"
    )
    
    g.add_edge("QUESTIONER", END)
    g.add_edge("ORCHESTRATOR", "DELIVERY")
    g.add_edge("DELIVERY", END)
    
    return g.compile()

director_ai = build_director_graph()