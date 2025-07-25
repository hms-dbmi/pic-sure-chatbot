"""
count.py

This module is responsible for interpreting user queries related to COUNT operations
and executing them against the PIC-SURE API. It includes two main steps:

1. `extract_count_query_filtering`: Uses an LLM to extract structured filters and genomic fields
   from a user question. It builds a tailored prompt using variable metadata and conversational context.

2. `count_from_metadata`: Converts LLM-filtered results into a properly formatted PIC-SURE COUNT
   query, handling:
   - dataset inference,
   - concept path matching,
   - genomic filter formatting,
   - API payload construction,
   - and final query execution.
"""

import json  # For encoding/decoding JSON data
import requests  # For sending HTTP requests (API calls)
from collections import Counter  # Utility for counting hashable items (e.g., value frequencies)
from pathlib import Path # For handling file paths in a platform-independent way
import yaml # For loading YAML configuration files (prompt templates)

from utils.llm import call_bedrock_llm, robust_llm_json_parse, format_variable_context, correct_filter_values

# Load all prompt templates from the YAML file located in the 'prompts' directory
with open(Path(__file__).parent.parent / "prompts" / "base_prompts.yml", "r") as f:
    PROMPTS = yaml.safe_load(f)  # Parse the YAML content into a Python dictionary

count_filter_extraction_prompt = PROMPTS["count_filter_extraction"]["system"]["prompt"]


def extract_count_query_filtering(user_question: str, variables: list, previous_interaction: tuple[str, str] = None) -> dict:
    """
    Calls a LLM to identify precise filters and optional genomic fields from the user's question.

    Args:
        user_question (str): The original user query (e.g., "How many women over 50 have diabetes?")
        variables (list): List of variables extracted from the data dictionary.
        previous_interaction (tuple[str, str], optional): Previous user question and bot answer for context.

    Returns:
        dict: A dictionary with:
              {
                "filters": {
                    "AGE": {"min": "50"},
                    "SEX": ["Female"]
                },
                "genomic": {
                    "gene": ["GRIN2A"]
                }
              }
    """

    # Step 1 – Create prompt context with all candidate variables
    variable_context = "Here are the candidate variables from which you can select filters:\n\n"
    variable_context += format_variable_context(variables)

    # Step 2 – Inject into prompt template
    prompt = count_filter_extraction_prompt.format(
        user_question=user_question,
        variable_context=variable_context
    )

    # Step 3 – If there's a previous interaction, prepend as conversational context
    if previous_interaction:
        previous_user, previous_bot = previous_interaction
        system_context = (
            f"The user previously asked:\nUser: {previous_user}\n"
            f"The assistant responded:\nAssistant: {previous_bot}\n\n"
            "Now the user asks a follow-up question. Interpret it in the context of the previous exchange.\n\n"
        )
        prompt = system_context + prompt

    # Step 4 – Send to LLM and validate output
    response = call_bedrock_llm(prompt=prompt)

    count_schema = {
        "filters": dict,
        "genomic": (dict, type(None))
    }
    fallback = {
        "filters": {},
        "genomic": None
    }

    parsed = robust_llm_json_parse(response, count_schema, fallback, label="Count filtering")

    return parsed


def count_from_metadata(
    user_question: str,
    metadata: dict,
    variables: list,
    token: str,
    api_url: str,
    previous_interaction: tuple[str, str] = None
):
    """
    Builds and sends a COUNT query to the PIC-SURE API based on user-intent metadata.
    It matches clues to variables, constructs the proper filter structure (both clinical 
    and genomic), and sends a synchronous COUNT request to PIC-SURE.

    Steps:
    - Use LLM to extract filters/genomic info.
    - Infer dataset if not specified.
    - Match variables to concept paths.
    - Format query payload.
    - Execute and return result.

    Args:
        user_question (str): The user's question.
        metadata (dict): Metadata including search terms, filter clues, and dataset.
        variables (list): List of all candidate variables.
        token (str): Bearer token for PIC-SURE authentication.
        api_url (str): Base URL for PIC-SURE API.
        previous_interaction (tuple[str, str], optional): Previous user question and bot answer for context.

    Returns:
        int or None: The resulting count from the PIC-SURE query, or None on failure.
    """

    # -------------------------------------------
    # Step 1 – Extract filters from the user question
    # -------------------------------------------
    print("\n🤖 Calling extract_count_query_filtering...")

    filtering_result = extract_count_query_filtering(
        user_question=user_question,
        variables=variables,
        previous_interaction=previous_interaction
    )

    print("🔎 LLM filtering result:", json.dumps(filtering_result, indent=2))

    filters = filtering_result.get("filters", {})
    genomic = filtering_result.get("genomic", None)
    dataset = metadata.get("dataset")

    # -------------------------------------------
    # Step 1.5 – Try to infer dataset from filtered variables if not given
    # -------------------------------------------
    if not dataset:
        # 1. Retrieves variables corresponding exactly to the filters
        matched_vars = [var for var in variables if var["name"] in filters and var.get("dataset")]

        # 2. Checks if they all belong to the same dataset
        dataset_set = set(var["dataset"] for var in matched_vars)
        if len(dataset_set) == 1:
            dataset = dataset_set.pop()
            print(f"✅ All filtered variables are from dataset: {dataset}")
        elif len(dataset_set) > 1:
            # 3. if not, select the most frequent dataset
            dataset_counts = Counter(var["dataset"] for var in matched_vars)
            dataset = dataset_counts.most_common(1)[0][0]
            print(f"⚠️ Multiple datasets found. Selected most common: {dataset}")
            # 4. Remove variables that are not in this dataset
            filters = {
                k: v for k, v in filters.items()
                if any(var["name"] == k and var["dataset"] == dataset for var in matched_vars)
            }
        else:
            print("❌ Could not infer dataset from filters.")

    print(f"\n📊 COUNT query for dataset: {dataset}")

    # -------------------------------------------
    # Step 2 – Clean filters using metadata constraints (e.g., chosen dataset, valid values, min/max bounds)
    # -------------------------------------------
    filters = correct_filter_values(filters, variables, dataset)
    print(f"✅ Corrected filters: {json.dumps(filters, indent=2)}")

    # Optional: deduplicate categorical values
    for k in filters:
        if isinstance(filters[k], list):
            filters[k] = list(set(filters[k]))

    # -------------------------------------------
    # Step 3 – Match variable names to their concept paths (required for API query)
    # -------------------------------------------
    filtered_vars = {}
    for v in variables:
        if v.get("dataset") == dataset:
            for f in filters:
                if v["name"].lower() == f.lower():
                    filtered_vars[f] = v["conceptPath"]

    print("✅ Matched variable names to concept paths:")
    for name, path in filtered_vars.items():
        print(f"  • {name} → {path}")

    # Warn if filters exist but no matching variable
    for name in filters:
        if not any(name.lower() == k.lower() for k in filtered_vars):
            print(f"⚠️ Variable '{name}' not found in dataset '{dataset}'.")

    # -------------------------------------------
    # Step 4 – Format genomic filters if present
    # -------------------------------------------
    variant_filters = []

    if genomic and isinstance(genomic, dict) and "gene" in genomic:
        genes = genomic["gene"]
        print(f"🧬 Genomic gene filter: {genes}")
        if isinstance(genes, list) and genes:
            variant_filters = [{
                "categoryVariantInfoFilters": {"Gene_with_variant": genes},
                "numericVariantInfoFilters": {}
            }]
    else:
        # Always provide the structure even if empty (required by PIC-SURE)
        variant_filters = [{
            "categoryVariantInfoFilters": {},
            "numericVariantInfoFilters": {}
        }]

    # -------------------------------------------
    # Step 5 – Abort if no clinical or genomic filters are applied
    # -------------------------------------------
    no_clinical = not filters
    no_genomic = all(
        not vf["categoryVariantInfoFilters"] and not vf["numericVariantInfoFilters"]
        for vf in variant_filters
    )
    if no_clinical and no_genomic:
        print("❌ No filters were applied. Aborting query.")
        return None

    # -------------------------------------------
    # Step 6 – Format API filters into payload structure
    # -------------------------------------------
    category_filters = {}
    numeric_filters = {}

    for name, condition in filters.items():
        path = filtered_vars.get(name)
        if not path:
            continue
        if isinstance(condition, list):
            # Categorical filter: list of accepted values
            category_filters[path] = condition
        elif isinstance(condition, dict):
            # Continuous filter: min/max bounds (as strings)
            numeric_filters[path] = {k: str(v) for k, v in condition.items() if v != ""}

    query = {
        "expectedResultType": "COUNT",
        "categoryFilters": category_filters,
        "numericFilters": numeric_filters,
        "fields": [],
        "anyRecordOf": [],
        "requiredFields": [],
        "variantInfoFilters": variant_filters
    }

    # -------------------------------------------
    # Step 7 – Send request to PIC-SURE API
    # -------------------------------------------
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "query": query,
        "resourceUUID": "ded89b08-faa9-435c-b7c4-55b81922ee5f"
    }

    print("📡 Sending COUNT query...")
    print(json.dumps(payload, indent=2))

    response = requests.post(
        url=f"{api_url}/query/sync",
        headers=headers,
        data=json.dumps(payload)
    )

    print(f"🔁 Status code: {response.status_code}")

    if response.status_code != 200:
        print(f"❌ Error {response.status_code}:\n{response.text}")
        return None

    # -------------------------------------------
    # Step 8 – Parse and return the response
    # -------------------------------------------
    try:
        result = response.json()
        if isinstance(result, int):
            print(f"✅ Success: {result}")
            return result
        else:
            print(f"❌ Unexpected response format: {result}")
            return None

    except Exception as e:
        print(f"❌ Failed to parse response: {e}")
        return None
