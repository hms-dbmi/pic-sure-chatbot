# -------------------------------------------------------
# API Chatbot Main Script (No RAG, only metadata queries)
# -------------------------------------------------------

import json
from collections import deque
from typing import List, Dict, Any

# Import utility functions for each intent type
from utils.extract_metadata import (
    extract_query_metadata,         # STEP 1: parse user question into structured metadata
    get_variables_from_metadata,    # STEP 2: fetch variables from datadictionary
)
from utils.count import count_from_metadata  # Intent: 'count' → numeric count query
from utils.distribution import (
    extract_distribution_dataframe, # Intent: 'distribution' → returns a DataFrame
    plot_distribution,              # saves the histogram
)
from utils.information import information_from_metadata  # Intent: 'metadata' → summary answer

from confidential import PICSURE_TOKEN, PICSURE_API_URL # Secure credentials


# --------------------------------------------
# Constants
# --------------------------------------------

MAX_HISTORY_LENGTH = 6 # Nombre maximum de messages dans l'historique de la conversation


# --------------------------------------------
# Chatbot Class
# --------------------------------------------

class APICallChatbot:
    """
    A chatbot that uses structured metadata extraction and PIC-SURE API calls
    to respond to clinical research questions without retrieval-augmented generation (RAG).
    """

    def __init__(self, token: str, api_url: str):
        """
        Initialize the chatbot with API credentials and an internal memory.

        Args:
            token (str): PIC-SURE authentication token.
            api_url (str): Base URL for the PIC-SURE API.
        """
        self.token = token
        self.api_url = api_url
        self.chat_history = deque(maxlen=MAX_HISTORY_LENGTH) # Limited rolling memory
        self.previous_interaction = None # Stores last user/bot turn for LLM context
        self.previous_extracted_metadata = None  # dict from previous extract_query_metadata


    def run_pipeline(self, user_question: str) -> str:
        """
        Core function: processes a user question through metadata extraction,
        variable lookup, and query execution depending on intent (count/distribution/metadata).

        Args:
            user_question (str): The user's natural language query.

        Returns:
            str: Final answer (text or filename), ready to display to the user.
        """

        # -----------------------------
        # Step 1: Extract metadata from the question
        # -----------------------------
        previous_user, previous_bot = None, None
        if self.previous_interaction:
            previous_user, previous_bot = self.previous_interaction

        metadata = extract_query_metadata(
            user_question,
            previous_interaction=(previous_user, previous_bot),
            previous_extracted_metadata=self.previous_extracted_metadata
        )


        # -----------------------------
        # Step 2: Retrieve relevant variables from the dictionary
        # -----------------------------
        variables = get_variables_from_metadata(metadata, self.token)
        print(f"Variables extracted: {variables}\n")

        # -----------------------------
        # Step 3: Route based on intent
        # -----------------------------
        if metadata["intent"] == "count":
            # → Handle COUNT request
            count_result = count_from_metadata(user_question, metadata, variables, self.token, self.api_url, self.previous_interaction)
            if count_result is not None:
                final_answer = f"There are {count_result} patients matching the criteria."
            else:
                final_answer = "❌ No filters could be applied — the query was skipped to avoid returning total population."


        elif metadata["intent"] == "distribution":
            # → Handle DISTRIBUTION request
            df, filter_desc = extract_distribution_dataframe(user_question, metadata, variables, self.token, self.api_url, self.previous_interaction)
            if df is not None and not df.empty:
                print("📊 Displaying distribution plot...")
                filename = plot_distribution(df, filter_description=filter_desc)

                if filename:
                    final_answer = f"The distribution was saved as '{filename}'."
                else:
                    final_answer = (
                        "⚠️ Unable to generate a distribution plot: "
                        "no valid numeric values found in the selected variable."
                    )

            else:
                print("⚠️ No data returned for selected variable.") 
                final_answer = "⚠️ No data available to plot the distribution."

        else:
            # → Handle information request (e.g., variable list, dataset suggestions)
            final_answer = information_from_metadata(user_question, variables, list(self.chat_history))


        # -----------------------------
        # Step 4: Update conversation memory
        # -----------------------------
        self.chat_history.append(f"User: {user_question}")
        self.chat_history.append(f"Bot: {final_answer}")
        self.previous_interaction = (user_question, final_answer)
        self.previous_extracted_metadata = metadata

        return final_answer

    def display_chat_history(self):
        """
        Utility function to print the full conversation history.
        """
        print("\n🧠 Chat History:")
        for line in self.chat_history:
            print(line)


# --------------------------------------------
# Interactive usage (CLI loop)
# --------------------------------------------
if __name__ == "__main__":
    chatbot = APICallChatbot(token=PICSURE_TOKEN, api_url=PICSURE_API_URL)

    print("🤖 Welcome to the API Chatbot (no RAG). Type 'exit' to quit.\n")
    while True:
        question = input("You: ")
        if question.strip().lower() == "exit":
            break
        response = chatbot.run_pipeline(question)
        print(f"Bot answer: {response}\n\n")
