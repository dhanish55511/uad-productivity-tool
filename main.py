from rag_pipeline import get_or_create_rag_data, retrieve_context
from agent_setup import setup_agents, initiate_agent_chat
from config import user_task, rag_query

if __name__ == "__main__":
    # 1. RAG Initialization
    # This call loads the embedding model and the FAISS index/documents
    get_or_create_rag_data()

    # 2. Context Retrieval
    # Manually call the retrieval function to get the context before the chat starts
    print(f"--- Pre-retrieving context for query: '{rag_query}' ---")
    initial_context = retrieve_context(rag_query, top_k=3)

    # 3. Agent Setup
    user_proxy, writer_agent, reviewer_agent, manager = setup_agents()

    # 4. Build the combined message (Task + Context)
    combined_initial_message = f"""
**Task:**
{user_task}

**CONTEXT:**
{initial_context}
"""
    
    # 5. Initiate the Chat
    # The agents will now take over, draft, and review the document
    initiate_agent_chat(manager, user_proxy, combined_initial_message)