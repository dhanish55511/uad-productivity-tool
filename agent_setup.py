import autogen
from config import (
    llm_config, 
    writer_system_prompt, 
    reviewer_system_prompt
)
from typing import Tuple, Any

def setup_agents() -> Tuple[autogen.UserProxyAgent, autogen.AssistantAgent, autogen.AssistantAgent, autogen.GroupChatManager]:
    """
    Initializes and configures the Writer Agent, Reviewer Agent, and the GroupChat Manager.
    """
    
    # AGENT 1: The Writer
    writer_agent = autogen.AssistantAgent(
        name="Writer_Agent",
        llm_config=llm_config,
        system_message=writer_system_prompt,
    )

    # AGENT 2: The Reviewer
    reviewer_agent = autogen.AssistantAgent(
        name="Reviewer_Agent",
        llm_config=llm_config,
        system_message=reviewer_system_prompt,
    )

    # USER PROXY: The "User" and Coordinator
    # Terminates chat when "APPROVED" is detected from the Reviewer
    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        human_input_mode="TERMINATE", 
        max_consecutive_auto_reply=10,
        code_execution_config=False,
    )

    # We create a group chat and manager
    groupchat = autogen.GroupChat(
        agents=[user_proxy, writer_agent, reviewer_agent], 
        messages=[], 
        max_round=15,
        # Forces a predictable turn order (Writer -> Reviewer -> Writer -> ...) 
        # to prevent the LLM manager from looping
        speaker_selection_method="round_robin" 
    )
    
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    
    print("Agents and GroupChat Manager configured.")

    return user_proxy, writer_agent, reviewer_agent, manager

def initiate_agent_chat(manager: autogen.GroupChatManager, user_proxy: autogen.UserProxyAgent, combined_message: str) -> Any:
    """
    Starts the multi-agent conversation.
    """
    print("--- Starting Agent Chat ---")
    chat_result = user_proxy.initiate_chat(
        manager,
        message=combined_message
    )
    print("--- Agent Chat Finished ---")
    return chat_result