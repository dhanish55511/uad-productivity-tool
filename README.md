## UAD Productivity Tool using Gen AI (RAG + AutoGen + Ollama)

This project implements a self-correcting, two-agent system designed to automatically draft and review user-facing documentation based on an internal functional specification (PDF).

The code has been modularized into four separate files for improved readability, maintainability, and organization.

### Key Technologies

RAG Pipeline: sentence-transformers (for embeddings) and faiss (for vector storage).

LLM Engine: Ollama running the llama3 model.

Agent Orchestration: PyAutoGen.

### Project Structure

The project is now organized into specialized files:

```text
uad-productivity-tool/
├── main.py                   # Main entry point (run this file)
├── config.py                 # All variables, paths, LLM settings, and prompts
├── rag_pipeline.py           # Functions for PDF processing, embedding, and retrieval
├── agent_setup.py            # Functions for defining the AutoGen agents and chat manager
├── requirements.txt          # List of Python dependencies
├── faiss_index.idx           # (Generated on first run) FAISS vector store
├── documents.json            # (Generated on first run) Text chunks from PDF
└── [YOUR_INPUT_PDF].pdf      # The Functional Specification PDF (see path in config.py)
```

### Prerequisites

Python 3.10+

Ollama: Must be installed and running on your machine.

Download from the official Ollama website.

Ensure the llama3 model is available locally by running:

ollama run llama3


PDF File: Your functional specification PDF must be accessible at the path defined in config.py.

### Installation

Clone this repository (or create the files).

Install the required Python libraries:

pip install -r requirements.txt


### Setup

1. Configure PDF Path

    Open config.py and modify the PDF_PATH variable to point to your document:

    > config.py (Line ~4)
    PDF_PATH = "C:/path/to/your/functional-spec.pdf" 
    > Example: PDF_PATH = "./functional-spec.pdf"


2. Verify Ollama

    Ensure your Ollama server is running and accessible on http://localhost:11434.

    #### How to Run

        Execute the main.py script:

        python main.py


    #### Workflow

        1. Initialization (main.py): Calls rag_pipeline.py to load data and agent_setup.py to define the system.

        2. RAG: Context is retrieved and combined with the user_task into a single message.

        3. Chat Start: The chat begins, forcing a reliable round-robin turn:

        4. Writer Agent (Writer_Agent): Drafts the structured documentation.

        5. Reviewer Agent (Reviewer_Agent): Reviews the draft against the Microsoft Style Guide.

        6. Termination: The chat automatically terminates when the Reviewer_Agent replies with the word "APPROVED".