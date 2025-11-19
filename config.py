from typing import Dict

# --- RAG and Path Configuration ---
# NOTE: Update the PDF_PATH to your actual file location
# PDF_PATH = "C:/Users/Dhanish Kumar/Documents/DhanishKr/UAD Assistance/Agent AI/ollama agent/input"
PDF_PATH = "C:/Users/Dhanish Kumar/Documents/Oracle_DhanishKr/UAD Assistance/Agent AI/ollama agent/Windows_Client" \
".pdf" # Use your PDF path
FAISS_INDEX_PATH = "faiss_index.idx"
DOCUMENTS_PATH = "documents.json"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
OLLAMA_MODEL = "llama3"

# --- LLM Configuration for AutoGen ---
# ASSUMPTION: Ollama is running at http://localhost:11434
llm_config: Dict = {
    "config_list": [
        {
            "model": OLLAMA_MODEL,
            "base_url": "http://localhost:11434/v1",  # Standard Ollama API endpoint
            "api_key": "ollama",  # Placeholder, but required
        },
    ],
    "cache_seed": None,
    "temperature": 0.5,  # Increased to encourage more robust, creative responses
}

# --- Agent System Prompts ---
writer_system_prompt = """
You are a senior technical writer. Your job is to draft a professional, user-facing document.
You will be given a task and a 'CONTEXT' block of text from a functional specification.
You MUST use ONLY the provided CONTEXT to complete the task. Do not make up information.

Your process is:
1.  Receive the initial task and context.
2.  Based *only* on the retrieved context, write a clear, user-friendly draft following the required structure.
3.  Pass the draft to the Reviewer.
4.  If the Reviewer provides feedback, revise the draft using the feedback and the original context.
"""

reviewer_system_prompt = """
You are a meticulous editor. Your ONLY job is to review the draft from the Writer_Agent.
You must ensure it follows the Microsoft Style Guide principles.
Key rules to check:
1.  **Active Voice:** Prefer active voice ("You can use..." not "It can be used...").
2.  **Clarity:** Is it easy to understand? Is there internal jargon?
3.  **Conciseness:** Is it wordy?
4.  **Tone:** Is it professional and helpful?

Provide specific, actionable feedback, referencing the text.
If the draft is good and meets all guidelines, respond with ONLY the word "APPROVED".
"""

# --- Initial Task and Query ---
user_task = """
Using the provided CONTEXT, draft a professional, user-facing documentation in detailed Markdown format.
The document MUST adhere to a realistic structure, including:
1.  A main H1 Title, using the name of the feature (e.g., # AI Results Guide).
2.  A concise, high-level introduction explaining the feature's primary benefit and purpose (The 'What').
3.  Organize the rest of the content into logical, descriptive H2 and H3 subheadings (e.g., 'Key Features and Capabilities', 'Requirements for Use', or 'Troubleshooting').
4.  The content within the sections must fully explain what the feature is, how it works, and for whom it is intended, using clear, active language (as per the Reviewer's style guide).
5.  All headings must be descriptive, functional, and summarize the content of their section.
"""

rag_query = "AI results feature"