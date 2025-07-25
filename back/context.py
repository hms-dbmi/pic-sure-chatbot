"""
context.py

This module provides helper functions to retrieve contextual information 
used throughout the PIC-SURE LLM pipeline.
In particular, it includes logic for dynamically fetching the list of datasets available from the PIC-SURE API.

Although not used dynamically in production (for now), this function is useful 
to refresh the known list of datasets that can be used for filtering or exploration.

Typical use case:
- Populate the list of datasets from PIC-SURE during a setup phase
- Manually define or store it in a configuration if API calls are restricted
"""

import requests
import json

PICSURE_DICT_URL = "https://nhanes.hms.harvard.edu/picsure/proxy/dictionary-api/concepts"

# Default number of lines to display per page (used for paginated output)
PAGE_SIZE = 55  # Number of variables per page for pagination

def get_available_datasets(token: str, max_pages: int = 50) -> list:
    """
    Dynamically fetch available datasets from the PIC-SURE dictionary.

    This function queries the PIC-SURE API to retrieve all unique datasets
    by paginating through the dictionary endpoint. It's typically run once 
    during initialization or debugging to discover which datasets are accessible.

    Args:
        token (str): PIC-SURE API authentication token (Bearer token).
        max_pages (int): Maximum number of pagination requests to perform 
                         (default = 50, enough for most use cases).

    Returns:
        list: Alphabetically sorted list of dataset names (e.g., ["Nhanes", "Synthea", "1000Genomes"]).
    
    Notes:
        - If the endpoint fails or the response is malformed, the function logs the error and exits gracefully.
        - If the dictionary is exhausted before `max_pages`, pagination stops early.
    """

    # Step 1 – Prepare headers and empty container
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    datasets = set()

    # Step 2 – Loop through dictionary pages to collect all dataset names
    for page_number in range(max_pages):
        params = {"page_number": page_number, "page_size": PAGE_SIZE}
        payload = {"facets": [], "search": ""}

        response = requests.post(PICSURE_DICT_URL, headers=headers, params=params, data=json.dumps(payload))

        if response.status_code != 200:
            print(f"❌ Error {response.status_code} while fetching datasets.")
            break

        try:
            data = response.json()
        except Exception as e:
            print(f"❌ Failed to parse JSON response: {e}")
            break

        # Step 3 – Extract dataset info from each variable
        variables = data.get("content", [])
        if not variables:
            break # End of data

        for var in variables:
            dataset = var.get("dataset") or var.get("dataset_id")
            if dataset:
                datasets.add(dataset)

        # Step 4 – Stop early if fewer results than a full page
        if len(variables) < PAGE_SIZE:
            break  # No more pages

    # Step 5 – Return sorted list
    return sorted(datasets)