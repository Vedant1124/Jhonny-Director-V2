import streamlit as st
import os
import sys
from PIL import Image

# 1. Page Config (MUST BE FIRST COMMAND)
st.set_page_config(page_title="JHONNY Director's Studio", page_icon="🎬", layout="wide")

# 2. Backend Import Setup
sys.path.append(os.getcwd())
try:
    from app import director_ai
except ImportError:
    st.error("⚠️ Backend Error: Could not import 'director_ai' from 'app.py'. Please ensure both files are in the same directory.")
    st.stop()

from langchain_core.messages import HumanMessage, AIMessage

# ============================================================
# 3. SESSION STATE MANAGEMENT
# ============================================================
# Stores the chat history
if "messages" not in st.session_state: 
    st.session_state.messages = []

# Stores the uploaded image path for the backend
if "image_path" not in st.session_state: 
    st.session_state.image_path = None

# CRITICAL: Persistence for the sequential interview
if "current_phase" not in st.session_state: 
    st.session_state.current_phase = "INIT"

if "collected_data" not in st.session_state: 
    st.session_state.collected_data = {}

# ============================================================
# 4. SIDEBAR: VISUAL INGESTION
# ============================================================
with st.sidebar:
    st.title("📸 Visual Context")
    st.caption("Upload a reference image to ground Jhonny's suggestions.")
    
    uploaded_file = st.file_uploader("Upload Scene / Product", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        # Save locally for BLIP analysis
        img_path = "temp_vision_input.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.image_path = img_path
        st.image(Image.open(uploaded_file), caption="Vision Model Active", use_container_width=True)
        st.success("Image cached for analysis.")
    else:
        st.info("No image? Jhonny will visualize from scratch.")

    st.divider()
    
    # Debug View (Optional - Helps you see the phase tracking)
    with st.expander("🛠️ System State (Debug)"):
        st.write(f"**Phase:** {st.session_state.current_phase}")
        st.json(st.session_state.collected_data)
        if st.button("Reset Conversation"):
            st.session_state.messages = []
            st.session_state.current_phase = "INIT"
            st.session_state.collected_data = {}
            st.session_state.image_path = None
            st.rerun()

# ============================================================
# 5. MAIN CHAT INTERFACE
# ============================================================
st.title("🎬 JHONNY: Senior Creative Director")
st.markdown("---")

# Render History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Handle User Input
if prompt := st.chat_input("Say 'Hi' to start..."):
    
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("JHONNY is thinking..."):
            
            # Prepare LangChain message history
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" 
                else AIMessage(content=m["content"]) 
                for m in st.session_state.messages
            ]
            
            # INVOKE THE GRAPH
            # We pass the persistent state variables to the backend
            result = director_ai.invoke({
                "messages": history,
                "image_path": st.session_state.image_path,
                "current_phase": st.session_state.current_phase,
                "collected_data": st.session_state.collected_data
            })
            
            # 3. UPDATE STATE FROM BACKEND RESPONSE
            # This is crucial: we sync the backend's progress back to the frontend
            st.session_state.current_phase = result.get("current_phase", "INIT")
            st.session_state.collected_data = result.get("collected_data", {})
            
            # 4. Display Output
            response_text = result.get("final_output", "Director Jhonny is thinking...")
            st.markdown(response_text)
            
            # Add to history
            st.session_state.messages.append({"role": "assistant", "content": response_text})

# Auto-scroll adjustment (Streamlit handles this natively usually)