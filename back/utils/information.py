"""
information.py - Handles metadata-driven user questions (intent = "information")

This module contains the logic used when a user's question is not about computing a count
or a distribution, but rather about understanding what variables or metadata are available
in the dataset. For instance, questions such as:

    • "What demographic variables are available?"
    • "Can I filter on ethnicity or income?"
    • "What kind of genomic information do you have?"

In these cases, the chatbot generates a natural language response based on the clinical
metadata (variable names, categories, concepts, etc.) using a foundation model.

Function:
---------
- information_from_metadata: 
    Formats variable metadata and uses a LLM to generate a user-friendly response,
    optionally incorporating prior chat history for better contextualization.
"""
import yaml # For loading YAML configuration files (prompt templates)
from pathlib import Path # For handling file paths in a platform-independent way

from utils.llm import call_bedrock_llm, format_variable_context

# Load all prompt templates from the YAML file located in the 'prompts' directory
with open(Path(__file__).parent.parent / "prompts" / "base_prompts.yml", "r") as f:
    PROMPTS = yaml.safe_load(f)  # Parse the YAML content into a Python dictionary

information_response_prompt = PROMPTS["information_response"]["system"]["prompt"]


def information_from_metadata(user_question: str, variables: list, chat_history=None):
    """
    Generates a clear, structured answer to a user's question using a LLM,
    based solely on the metadata of available clinical variables.

    This function is triggered when the user's intent is to explore available variables,
    datasets, or metadata (e.g., "What demographic variables are available?").

    Args:
        user_question (str): The original question asked by the user.
        variables (list): A list of variable dictionaries returned by the dictionary API.
        chat_history (list[str], optional): Optional conversation history for context.

    Returns:
        str: A well-formatted and relevant natural language answer generated by the LLM.
    """

    # -------------------------------------------------
    # Step 1: Build a structured summary of variables
    # -------------------------------------------------
    if not variables:
        variable_context = "No variable metadata was found.\n"
    else:
        variable_context = "You have access to the following clinical variable metadata:\n"
        variable_context += format_variable_context(variables)

    # -------------------------------------------------
    # Step 2: Build the prompt to send to the LLM
    # -------------------------------------------------

    full_prompt = information_response_prompt.format(
        variable_context=variable_context,
        user_question=user_question
    )

    # -------------------------------------------------
    # Step 3: Include conversation history if provided
    # -------------------------------------------------
    if chat_history and len(chat_history) > 1:
        history_text = "\n".join(chat_history)
        full_prompt = (
            "Previous chat history (for context only):\n"
            f"{history_text}\n\n"
            "Now answer the new user question based on the variables below.\n\n"
        ) + full_prompt
    
    # -------------------------------------------------
    # Step 4: Call the LLM through Amazon Bedrock
    # -------------------------------------------------
    try:
        response = call_bedrock_llm(full_prompt, max_tokens=1000)
        return response

    except Exception as e:
        return f"❌ Failed to generate response: {e}"