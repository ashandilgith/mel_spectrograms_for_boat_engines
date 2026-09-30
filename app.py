import streamlit as st
import os
import json
import uuid
import base64
from openai import OpenAI
from dotenv import load_dotenv

# Import the core ML inference engine built previously
from system_1_classifier.inference import EngineDiagnosticInference

# Load environment variables (API Keys)
load_dotenv()

# --- THEME & CONFIGURATION ---
st.set_page_config(page_title="Piranaware Diagnostic Hub", layout="wide")

st.markdown("""
    <style>
    /* =========================================
       PIRANAWARE — PREMIUM MARINE INTERFACE
       Aesthetic changes only
       ========================================= */

    /* Global typography and background */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --background: #101c19;
        --surface: #182622;
        --surface-light: #20332d;
        --border: #30443c;
        --accent: #d6b66a;
        --accent-hover: #e6c77b;
        --text-primary: #f2f0e8;
        --text-secondary: #a8b6ad;
        --success: #83b99b;
    }

    .stApp {
        background-color: var(--background);
        color: var(--text-primary);
        font-family: 'DM Sans', sans-serif;
    }

    /* Main content spacing */
    .main .block-container {
        max-width: 1600px;
        padding: 3rem 4rem 4rem 4rem;
    }

    /* Headings */
    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.6px;
    }

    h1 {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem !important;
    }

    h2 {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--accent) !important;
        margin-bottom: 1rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
    }

    p, label, .stMarkdown {
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
        line-height: 1.7;
    }

    /* Secondary text */
    .stCaption, small {
        color: var(--text-secondary) !important;
    }

    /* Column alignment */
    [data-testid="column"] {
        padding: 0 0.75rem;
    }

    /* Section spacing */
    [data-testid="stVerticalBlock"] {
        gap: 1.2rem;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        transition: border-color 0.2s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent);
    }

    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
        border: none !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: var(--surface-light) !important;
        color: var(--accent) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: var(--accent) !important;
        color: var(--background) !important;
        border-color: var(--accent) !important;
    }

    [data-testid="stFileUploader"] small {
        color: var(--text-secondary) !important;
    }

    /* Primary buttons */
    .stButton > button {
        background-color: var(--accent) !important;
        color: #17211c !important;
        border: 1px solid var(--accent) !important;
        border-radius: 9px !important;
        padding: 0.65rem 1.2rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
        box-shadow: 0 4px 14px rgba(214, 182, 106, 0.15);
        transform: translateY(-1px);
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 2px var(--background),
                    0 0 0 4px var(--accent) !important;
    }

    /* Chat container */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--surface);
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 14px;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: var(--surface-light) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 1.1rem 1.25rem !important;
        margin-bottom: 0.8rem;
    }

    [data-testid="stChatMessage"] p {
        color: var(--text-primary) !important;
        line-height: 1.75;
    }

    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {
        background-color: var(--accent) !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    [data-testid="stChatInput"] textarea {
        background-color: var(--surface) !important;
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        border-radius: 12px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-secondary) !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent);
    }

    /* Other text inputs */
    .stTextInput input,
    .stTextArea textarea {
        background-color: var(--surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 9px !important;
    }

    /* JSON output */
    [data-testid="stJson"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
    }

    /* Success and error messages */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        border: 1px solid var(--border);
    }

    [data-testid="stAlert"] p {
        color: var(--text-primary) !important;
    }

    /* Spinner */
    [data-testid="stSpinner"] p {
        color: var(--accent) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #14211d;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {
        color: var(--text-primary) !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--background);
    }

    ::-webkit-scrollbar-thumb {
        background: #40534a;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent);
    }

    /* Responsive layout */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1.5rem 1rem 2rem 1rem;
        }

        h1 {
            font-size: 1.8rem !important;
        }

        h2 {
            font-size: 1.3rem !important;
        }

        [data-testid="column"] {
            padding: 0;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- LLM SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are the Piranaware Lead Marine Diagnostic Technician. Your role is to guide boat owners through complex engine failures. 
You are NOT a generic AI. You are a relentless, highly technical mechanic.

WORKFLOW RULES:
1. The system will invisibly pass you the JSON results from our Convolutional Neural Network's acoustic analysis.
2. DO NOT just list the results back to the user. Instead, use them as your starting hypothesis.
3. Immediately ask 1 or 2 highly targeted, probing questions to validate the ML model's findings (e.g., "The acoustic model detected a lean bog. Does this happen only when the engine is hot, or right off the dock?", or "The model flagged a belt squeal. Have you noticed rubber dust near the alternator?").
4. If the user uploads images (e.g., spark plugs, fuel filters, invoices), analyze them for wear, carbon fouling, or incorrect parts.
5. Synthesize the acoustic data, visual evidence, and user answers to narrow down the exact mechanical failure.
6. Provide concrete, step-by-step troubleshooting actions. Do not provide vague advice.
"""

# --- HELPER: IMAGE ENCODING ---
def encode_image(uploaded_file):
    """Encodes a Streamlit UploadedFile to base64 for OpenAI Vision."""
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

# --- INITIALIZATION ---
@st.cache_resource
def init_inference_engine():
    # Only loads the CNN once into memory
    return EngineDiagnosticInference()

if "openai_client" not in st.session_state:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY environment variable is missing. Set it in Codespaces Secrets or .env")
        st.stop()
    # OpenAI automatically picks up the OPENAI_API_KEY environment variable
    st.session_state.openai_client = OpenAI()

if "messages" not in st.session_state:
    # State tracking. "hidden" key ensures system instructions and background API data aren't rendered in the UI.
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT, "hidden": True},
        {"role": "assistant", "content": "Welcome to Piranaware. Upload your engine audio on the left to begin the acoustic scan, or describe your issue below.", "hidden": False}
    ]

try:
    inferencer = init_inference_engine()
    inference_ready = True
except Exception as e:
    inference_ready = False
    st.sidebar.error(f"⚠️ CNN Model not loaded. Ensure you have trained the model first. Error: {e}")

# --- API FORMATTER ---
def get_api_messages():
    """Strips UI-only keys (like 'hidden') before sending the array to the OpenAI API."""
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

# --- UI LAYOUT ---
st.title("Piranaware Triage & Diagnostic Hub")

col1, col2 = st.columns([1, 2], gap="large")

# --- LEFT COLUMN: ACOUSTIC INFERENCE ---
with col1:
    st.header("1. Acoustic Scan")
    st.write("Upload a `.wav` file of your engine stalling, squealing, or misfiring.")
    
    audio_file = st.file_uploader("Engine Audio Upload", type=["wav"])
    
    if audio_file and inference_ready:
        if st.button("Run Deep Learning Inference", use_container_width=True):
            with st.spinner("Analyzing mel-spectrograms..."):
                # Save to safe temporary path to prevent Docker/Codespace lock errors
                temp_path = f"/tmp/{uuid.uuid4()}_{audio_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(audio_file.getbuffer())
                
                # Execute CNN inference
                results = inferencer.analyze_recording(temp_path)
                os.remove(temp_path)
                
                if "error" in results:
                    st.error(results["error"])
                else:
                    st.success("Acoustic Analysis Complete")
                    st.json(results)
                    
                    # INVISIBLE HANDOFF TO LLM
                    handoff_msg = f"SYSTEM INJECTION: The acoustic scan just completed. Here is the raw JSON output from the CNN: {json.dumps(results)}. Do not say 'I received the JSON'. Immediately address the user, state what the model suspects, and ask your first probing question to isolate the root cause."
                    
                    # Append silently to state
                    st.session_state.messages.append({"role": "user", "content": handoff_msg, "hidden": True})
                    
                    # Trigger OpenAI inference
                    response = st.session_state.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=get_api_messages(),
                        temperature=0.3
                    )
                    
                    reply = response.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": reply, "hidden": False})
                    st.rerun()

# --- RIGHT COLUMN: MULTIMODAL CHAT & DIAGNOSTICS ---
with col2:
    st.header("2. Diagnostic Investigation")
    
    # Display Chat History
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            if msg.get("hidden"):
                continue
            with st.chat_message(msg["role"]):
                # Handle complex multimodal payload rendering (extracting just the text for UI)
                if isinstance(msg["content"], list):
                    text = next((item["text"] for item in msg["content"] if item.get("type") == "text"), "")
                    st.markdown(text)
                else:
                    st.markdown(msg["content"])
    
    # Multimodal Attachments
    st.write("### Attach Evidence")
    uploaded_images = st.file_uploader("Upload photos of parts, spark plugs, or repair invoices (Optional)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    # User Input
    if prompt := st.chat_input("Answer the technician's questions or describe the engine behavior..."):
        
        # Prepare content payload. If images exist, format for GPT-4o Vision.
        if uploaded_images:
            user_content = [{"type": "text", "text": prompt}]
            for img_file in uploaded_images:
                base64_image = encode_image(img_file)
                mime_type = img_file.type
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                })
        else:
            user_content = prompt
        
        # Append to state and UI
        st.session_state.messages.append({"role": "user", "content": user_content, "hidden": False})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
                
        # Send to OpenAI
        with st.spinner("Technician is analyzing..."):
            response = st.session_state.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=get_api_messages(),
                temperature=0.3
            )
            
        # Append and display response
        reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": reply, "hidden": False})
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(reply)