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
