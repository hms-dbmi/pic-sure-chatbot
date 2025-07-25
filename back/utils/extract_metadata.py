"""
extract_metadata.py

This module contains all logic related to extracting structured query metadata
from a user's natural language question. It includes:

1. The use of an LLM to extract:
   - query intent (e.g., count, distribution, metadata)
   - dataset (e.g., "Synthea", "Nhanes")
   - search terms (keywords to match dictionary variables)
   - type of variables (categorical, continuous)

2. Dictionary variable retrieval logic using the extracted metadata, including:
   - pagination over the PIC-SURE API
   - filtering by dataset and search terms
   - conversion into a list of usable variables for later filtering or plotting

These tools are part of the early-stage logic in the PIC-SURE chatbot pipeline
and are reused across count, distribution, and metadata exploration steps.
"""

import json  # For encoding/decoding JSON data
import requests  # For sending HTTP requests (API calls)
import yaml # For loading YAML configuration files (prompt templates)
from pathlib import Path # For handling file paths in a platform-independent way

from utils.llm import call_bedrock_llm, robust_llm_json_parse

# URL for retrieving the PIC-SURE dictionary (concepts metadata)
PICSURE_DICT_URL = "https://nhanes.hms.harvard.edu/picsure/proxy/dictionary-api/concepts"

# List of datasets available (can be dynamically fetched later)
DATASETS = ["Nhanes", "Synthea", "1000Genomes"]

# Default number of lines to display per page (used for paginated output)
PAGE_SIZE = 55  # Can be adjusted 

with open(Path(__file__).parent.parent / "prompts" / "base_prompts.yml", "r") as f:
    PROMPTS = yaml.safe_load(f)  # Parse the YAML content into a Python dictionary

metadata_extraction_prompt = PROMPTS["metadata_extraction"]["system"]["prompt"]


def extract_query_metadata(
   user_question: str,
   previous_interaction: tuple[str, str] = None,
   previous_extracted_metadata: dict = None
) -> dict:
   """
   Extracts structured metadata from a user's natural language question using a LLM model.

   Uses a guided system prompt to request the LLM to return a structured object containing:
   - intent (str): e.g. "count", "distribution", "metadata"
   - search_terms (list[str]): extracted biomedical keywords
   - dataset (str or None): one of the supported dataset names
   - type (str or None): variable type ("categorical", "continuous", or "both")

   Optionally takes into account the conversational history (e.g. follow-up questions).

   Args:
      user_question (str): The user's current natural language query.
      previous_interaction (tuple[str, str], optional): The (user, assistant) last exchange.
      previous_extracted_metadata (dict, optional): Metadata from the last step for continuity.

   Returns:
      dict: Cleaned and validated metadata dictionary for downstream use.
   """

   # Build base prompt from the question and dataset list
   system_prompt = metadata_extraction_prompt.format(datasets=DATASETS, user_question=user_question)

   # Incorporate context if a previous interaction is provided
   if previous_interaction:
      previous_user, previous_bot = previous_interaction

      # Add prior question and metadata if available
      system_prompt = (
         f"The user previously asked:\nUser: {previous_user}\n"
         + (
               f"From the previous interaction, the assistant extracted the following metadata:\n"
               f"{json.dumps(previous_extracted_metadata, indent=2)}\n"
               "You should take this into account when interpreting the current question.\n"
               "In particular, preserve relevant metadata like search_terms, intent, dataset, etc.\n"
               "You must carry over the relevant context from the previous question. "
               "This includes keeping the same intent, dataset, search_terms,"
               "as long as they still apply to the current question.\n"
               "In particular, make sure to retain and update the `search_terms` carefully, "
               "as they are essential for retrieving relevant variables in biomedical datasets.\n"
               if previous_extracted_metadata else ""
         )
         + f"The assistant responded:\nAssistant: {previous_bot}\n\n"
         + "Now the user asks a follow-up question. Interpret it in context.\n\n"
         + system_prompt
      )

   # Log the context
   print(f"\nPrevious user question: {previous_interaction[0] if previous_interaction else 'N/A'}")
   print(f"Previous extracted metadata: {json.dumps(previous_extracted_metadata, indent=2) if previous_extracted_metadata else 'None'}")
   print(f"Previous bot answer: {previous_interaction[1] if previous_interaction else 'N/A'}")

   try:
      print(f"User question: {user_question} \n")

      # Call the LLM with system prompt
      llm_response = call_bedrock_llm(system_prompt, max_tokens=500)

      print("🔍 LLM Metadata raw response:", llm_response)

      # Define expected schema
      metadata_schema = {
         "intent": str,
         "search_terms": list,
         "dataset": (str, type(None)),
         "type": (str, type(None))
      }
      fallback = {
         "intent": "information",
         "search_terms": [],
         "dataset": None,
         "type": None
      }

      # Validate and correct LLM output
      parsed = robust_llm_json_parse(llm_response, metadata_schema, fallback, label="Query metadata")

      # Remove dataset names from search terms to avoid redundancy
      normalized_datasets = [ds.lower() for ds in DATASETS]
      cleaned_terms = [term for term in parsed.get("search_terms", []) if term.lower() not in normalized_datasets]
      parsed["search_terms"] = cleaned_terms

      # Normalize dataset name (if provided) based on known datasets list
      if parsed.get("dataset") is not None:
         dataset_lower = parsed["dataset"].lower()
         matched_dataset = next((ds for ds in DATASETS if ds.lower() == dataset_lower), None)
         if matched_dataset:
               parsed["dataset"] = matched_dataset
         else:
               print(f"⚠️ Dataset '{parsed['dataset']}' not recognized. Replacing with None.")
               parsed["dataset"] = None

      print("✅ LLM Metadata cleaned response:", parsed)
      return parsed

   except Exception as e:
      print(f"❌ Failed to extract metadata: {e}")
      return fallback


def get_dictionary_variables(token, dataset=None, data_type=None, search_term="", max_pages=100):
   """
   Queries the PIC-SURE dictionary API to retrieve variable metadata,
   filtered by dataset name, variable type, and search term.

   Args:
      token (str): PIC-SURE authentication token.
      dataset (str or None): Restrict search to a specific dataset.
      data_type (str or None): Variable type ("categorical" or "continuous").
      search_term (str): Keyword to search for in variable names/descriptions.
      max_pages (int): Max number of pages to fetch from paginated API.

   Returns:
      list: A list of variable dictionaries (concept metadata).
   """

   # Step 1 – Prepare headers for API authentication
   headers = {
      "Authorization": f"Bearer {token}",
      "Content-Type": "application/json"
   }

   # Step 2 – Build filter facets to restrict the search
   facets = []
   if dataset:
      facets.append({"category": "dataset_id", "name": dataset})
   # We intentionally skip data_type to avoid over-filtering results
      # if data_type:
   #     facets.append({"category": "data_type", "name": data_type})
   #TODO enlever type de extract_query_metadata, trop restrictif et pose des probleme. 

   all_results = []

   # Step 3 – Iterate over pages to collect all results (pagination)
   for page_number in range(max_pages):
      # Define request parameters and body
      params = {"page_number": page_number, "page_size": PAGE_SIZE}
      payload = {
         "facets": facets,
         "search": search_term
      }

      # Step 4 – Send POST request to the PIC-SURE dictionary endpoint
      try:
         response = requests.post(
               PICSURE_DICT_URL,
               headers=headers,
               params=params,
               data=json.dumps(payload)
         )
      except Exception as e:
         print(f"❌ Request error: {e}")
         break

      # Step 5 – Check HTTP response status and stop if the response is not successful
      if response.status_code != 200:
         print(f"❌ Error {response.status_code} on page {page_number}")
         print(f"📥 Response content: {response.text}")
         break

      # Step 6 – Parse the JSON response content
      try:
         data = response.json()
      except Exception as e:
         print(f"❌ JSON decode error: {e}")
         print(f"📥 Raw content: {response.text}")
         break

      # Step 7 – Extract variable metadata from current page
      results = data.get("content", [])
      print(f"✅ Results on page {page_number}: {len(results)}")

      # Stop pagination if empty response (last page reached)
      if not results:
         break

      all_results.extend(results)

      # Stop early if page is not fully populated (last batch)
      if len(results) < PAGE_SIZE:
         break

   print(f"📦 Total results aggregated: {len(all_results)}")

   return all_results


def get_variables_from_metadata(metadata: dict, token: str, max_pages: int = 100):
   """
   Fetches relevant variables from the PIC-SURE dictionary using extracted metadata.

   Based on:
   - search_terms: user-relevant keywords
   - dataset: selected dataset name
   - type: variable type (categorical/continuous/both)

   Args:
      metadata (dict): Metadata returned by `extract_query_metadata`.
      token (str): PIC-SURE API token.
      max_pages (int): Max pages to fetch for each search term.

   Returns:
      list: A combined list of all variables retrieved from the dictionary.
   """

   # Step 1 – Extract search parameters from metadata
   search_terms = metadata.get("search_terms", [])
   dataset = metadata.get("dataset")
   data_type = metadata.get("type")

   # Step 2 – Treat 'both' as no filtering on variable type
   if data_type == "both":
      data_type = None

   # Logging
   print(f"\n📦 Fetching variables using:")
   print(f"🔎 Search terms: {search_terms}")
   print(f"📚 Dataset: {dataset}")
   print(f"🔢 Variable type: {data_type}\n")

   all_results = []

   # Step 3 – Iterate over each search term independently
   for term in search_terms:
      print(f"\n🔍 Searching PIC-SURE for: '{term}'")
      try:
         # Call dictionary API for this term
         results = get_dictionary_variables(
               token=token,
               dataset=dataset,
               data_type=data_type,
               search_term=term,
               max_pages=max_pages
         )

         # Log success or warning if no match
         if results:
               print(f"✅ Found {len(results)} variable(s) for '{term}'")
         else:
               print(f"❌ No variables found (or API error) for '{term}'")

         # Merge results
         all_results.extend(results)

      # Step 4 – Catch any search failure per keyword
      except Exception as e:
         print(f"⚠️ Error while searching term '{term}': {e}")

   print(f"\n📦 Total variables found: {len(all_results)}\n")

   return all_results
