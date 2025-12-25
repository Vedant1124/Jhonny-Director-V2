# 🎬 Director Jhonny: AI Cinematic Orchestrator

**Director Jhonny** is an autonomous multi-agent system designed to act as a **Senior Creative Producer**. Unlike standard chatbots, Jhonny operates on a deterministic "Cinematic Physics" engine, strictly enforcing optical rules, lighting interactions, and platform-specific delivery standards (Netflix vs. Instagram).

Using a **LangGraph** state machine, Jhonny visually analyzes reference images, conducts a structured technical interview, and orchestrates production-ready shot lists without "hallucinating" physically impossible shots.

---

## 🚀 Key Features

* **👁️ Vision-Grounded Perception**: Uses **Salesforce BLIP** to "see" uploaded reference images, extracting lighting conditions, textures, and subjects to inform creative decisions.
* **🧠 Multi-Agent Architecture**:
    * **Supervisor Node**: The "Brain" that manages the state and identifies missing technical pillars (Lens, Platform, VFX, etc.).
    * **Questioner Node**: The "Voice" that conducts a polite, sequential interview to fill gaps without overwhelming the user.
    * **Orchestrator Node**: Applies 180M+ vector-embedded technical rules to generate the shot list.
* **📐 Deterministic "Cinematic Physics"**:
    * **Auto-Lens Logic**: Automatically maps emotions to focal lengths (e.g., *Tension* = 85mm, *Calm* = 35mm).
    * **Lighting Physics**: Adjusts lighting descriptions based on lens choice (e.g., Wide lenses soften shadows; Telephoto deepens contrast).
* **📦 Platform-Specific Delivery**: automatically configures aspect ratios, codecs, and color science for **Netflix (HDR, Rec.2020)** vs. **Instagram (9:16, High Contrast)**.
* **🛡️ Zero-Gap Protocol**: The system refuses to generate output until all mandatory production pillars are defined or the user explicitly invokes the "Let Jhonny Decide" auto-mode.

---

## 🛠️ Architecture

The project is built on a **State Graph** using `LangGraph`:

```mermaid
graph TD
    A[Start] --> B(Image Ingestion / Vision)
    B --> C{Supervisor / Analyst}
    C -- Missing Info --> D[Questioner Node]
    D --> E[User Input]
    E --> C
    C -- Brief Complete --> F[Master Orchestrator]
    F --> G[Delivery / Formatter]
    G --> H[End]


📂 Project Structure
Bash

├── app.py                 # Core Backend (LangGraph engine & logic)
├── streamlit_app.py       # Frontend UI (Image upload & Chat interface)
├── requirements.txt       # Python dependencies
├── .env                   # API Keys (Groq, HuggingFace)
└── data/                  # RAG Knowledge Base
    ├── Engine/            # FAISS Index: Supervisor Rules
    ├── Orchestrator/      # FAISS Index: Shot Logic & Physics
    ├── Images/            # FAISS Index: Visual Grounding Data
    └── A-N/               # FAISS Index: QA & Validation Rules
⚙️ Installation & Setup
1. Clone the Repository
Bash

git clone [https://github.com/yourusername/director-jhonny.git](https://github.com/yourusername/director-jhonny.git)
cd director-jhonny
2. Create a Virtual Environment
Bash

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash

pip install -r requirements.txt
(Note: Ensure you have PyTorch installed for the Transformers vision pipeline)

4. Configure Environment Variables
Create a .env file in the root directory:

Ini, TOML

GROQ_API_KEY=gsk_your_groq_api_key_here
HF_TOKEN=hf_your_huggingface_token_here
Groq API: Used for the ultra-fast Llama 3 reasoning engine.

HF Token: Used for the BLIP vision model fallback (optional if running local).

🚀 Usage
Run the Streamlit frontend to launch the Director's Studio:

Bash

streamlit run streamlit_app.py
Upload a Reference Image (Optional): Jhonny will analyze it to suggest lighting and lenses.

Start Chatting: Say "Hi". Jhonny will introduce himself and start the "Zero Gap" interview process.

The "Magic" Phrase: At any point, type "Let Jhonny Decide" to have the AI strictly apply its internal physics engine to autofill the remaining details.

🧠 The Logic Engines
Jhonny is powered by several RAG (Retrieval Augmented Generation) indexes:

Engine G (The Supervisor): Contains rules for interviewing and project management.

Merged-Multipillar (The Orchestrator): The massive index containing the "Physics" of filmmaking—how lenses affect light, how emotions map to camera motion, and platform delivery specs.

Image Master: Contains technical descriptions of visual concepts to ground the vision model.

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
Distributed under the MIT License. See LICENSE for more information.

🌟 Acknowledgements
LangChain & LangGraph for the stateful agent framework.

Groq for lightning-fast LPU inference.

Salesforce for the BLIP vision model.
