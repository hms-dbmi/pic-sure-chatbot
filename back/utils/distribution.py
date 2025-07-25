"""
distribution.py

This module handles the logic for interpreting and executing distribution-type queries
in the PIC-SURE pipeline. It includes:

1. LLM-based interpretation of the user's intent to:
   - identify the main variable to plot (must be continuous),
   - extract clinical and genomic filters,
   - determine the relevant fields to retrieve.

2. PIC-SURE API query generation to retrieve patient-level data
   for the selected variable and filters.

3. Histogram plotting from the resulting distribution.

Main functions:
- extract_distribution_metadata
- extract_distribution_dataframe
- plot_distribution
"""

import json  # For encoding/decoding JSON data
import pandas as pd  # For handling tabular data (DataFrames)
import numpy as np  # For numerical operations (arrays, histograms, etc.)
import matplotlib.pyplot as plt  # For plotting data visualizations
import requests  # For sending HTTP requests (API calls)
from io import StringIO
import yaml # For loading YAML configuration files (prompt templates)
from pathlib import Path # For handling file paths in a platform-independent way

from utils.llm import call_bedrock_llm, robust_llm_json_parse, format_variable_context, correct_filter_values

# Load all prompt templates from the YAML file located in the 'prompts' directory
with open(Path(__file__).parent.parent / "prompts" / "base_prompts.yml", "r") as f:
    PROMPTS = yaml.safe_load(f)  # Parse the YAML content into a Python dictionary

distribution_filter_extraction_prompt = PROMPTS["distribution_extraction"]["system"]["prompt"]

# Retrieves the path to the current folder (where the script is running)
current_dir = Path(__file__).resolve().parent
#Locate the "plots" folder in the current folder's parent folder
plot_dir = current_dir / "../plots"


def extract_distribution_metadata(user_question: str, variables: list, previous_interaction: tuple[str, str] = None) -> dict:
    """
    Uses an LLM to extract key components for a distribution query.

    Extracted:
    - The main variable to analyze (must be continuous),
    - Optional clinical filters (AGE, SEX...),
    - Names of all fields to retrieve from the patient-level database.
    - Optional genomic filters (e.g., genes),

    The LLM is expected to return variable names (not conceptPaths). Mapping to conceptPaths happens downstream.

    Args:
        user_question (str): The original user question (free text).
        variables (list): List of available variables (name, type, values, conceptPath, etc.).
        previous_interaction (tuple[str, str], optional): (previous_user_question, previous_bot_answer)

    Returns:
        dict: {
            "distribution_variable": str or null,           # name of the variable (not conceptPath)
            "filters": dict[str, list or dict],             # { "AGE": {"min": 50}, "SEX": ["Female"] }
            "fields": list[str] or [],                      # e.g., ["AGE", "SEX", "BMI"]
            "genomic": dict[str, list] or null              # { "gene": ["BRCA1"] }
        }
    """

    # Step 1 – Create prompt context using all available variables
    variable_context = "Here are the candidate variables from which you can select filters:\n\n"
    variable_context += format_variable_context(variables)

    # Step 2 – Inject into distribution prompt template
    prompt = distribution_filter_extraction_prompt.format(
        user_question=user_question,
        variable_context=variable_context
    )

    # Step 3 – Add conversational context if applicable
    if previous_interaction:
        previous_user, previous_bot = previous_interaction
        system_context = (
            f"The user previously asked:\nUser: {previous_user}\n"
            f"The assistant responded:\nAssistant: {previous_bot}\n\n"
            "Now the user asks a follow-up question. Interpret it in the context of the previous exchange.\n\n"
        )
        prompt = system_context + prompt

    # Step 4 – Call LLM and validate response
    response = call_bedrock_llm(prompt=prompt)

    distribution_schema = {
        "distribution_variable": str,
        "filters": (dict, type(None)),
        "fields": list,
        "genomic": (dict, type(None))
    }
    fallback = {
        "distribution_variable": None,
        "filters": {},
        "fields": [],
        "genomic": None
    }

    parsed = robust_llm_json_parse(response, distribution_schema, fallback, label="Distribution metadata")

    return parsed


def extract_distribution_dataframe(user_question: str, metadata: dict, variables: list, token: str, api_url: str, previous_interaction: tuple[str, str] = None) -> pd.DataFrame:
    """
    Extracts a patient-level DataFrame of values for a selected variable (usually continuous),
    applying clinical and genomic filters. The variable to plot is identified by name and mapped to its conceptPath.

    Steps:
    - Ask LLM to determine distribution variable, filters, and fields.
    - Identify matching variable and dataset.
    - Map names to conceptPaths.
    - Build API payload with filters and fields.
    - Send request to PIC-SURE.
    - Parse CSV and return relevant column.

    Args:
        user_question (str): Natural language user prompt.
        metadata (dict): Contains 'dataset' (optional).
        variables (list): List of variable dicts.
        token (str): API token.
        api_url (str): PIC-SURE API base URL.
        previous_interaction (tuple, optional): Prior (user, bot) exchange.

    Returns:
        Tuple[pd.DataFrame or None, str or None]: Filtered data + filter name descriptor.
    """

    # -------------------------------------------
    # Step 1 – Unpack LLM result
    # -------------------------------------------
    print("\n🤖 Calling extract_distribution_metadata...")
    filtering_result = extract_distribution_metadata(user_question, variables, previous_interaction)
    print("🔎 Raw LLM result:", json.dumps(filtering_result, indent=2))

    filters = filtering_result.get("filters", {})
    genomic = filtering_result.get("genomic", None)
    dist_var_name = filtering_result.get("distribution_variable")
    field_names = filtering_result.get("fields", [])

    # Fail early if LLM did not return a valid distribution variable
    if not dist_var_name:
        print("❌ No distribution variable provided.")
        return None, None

    # -------------------------------------------
    # Step 2 – Determine dataset and select the correct distribution variable
    # -------------------------------------------
    dataset = metadata.get("dataset")
    selected_variable = None

    if not dataset:
        # No dataset provided → search variable name across all datasets
        candidate_vars = [v for v in variables if v["name"].lower() == dist_var_name.lower()]
        if not candidate_vars:
            print(f"❌ No variable named '{dist_var_name}' found in any dataset.")
            return None, None
        
        # Heuristic: choose dataset with the most matching filters
        def filter_match_score(var):
            ds = var["dataset"]
            return sum(1 for f in filters if any(v["name"] == f and v["dataset"] == ds for v in variables))

        candidate_vars.sort(key=filter_match_score, reverse=True)
        selected_variable = candidate_vars[0]
        dataset = selected_variable["dataset"]
        print(f"📌 Dataset inferred from variable distribution + filters: {dataset}")

    else:
        # Dataset is predefined → just confirm the variable exists
        selected_variable = next(
            (v for v in variables if v["name"].lower() == dist_var_name.lower() and v["dataset"] == dataset),
            None
        )
        if not selected_variable:
            print(f"❌ Variable '{dist_var_name}' not found in dataset '{dataset}'.")
            return None, None

    concept_path = selected_variable["conceptPath"]
    var_name = selected_variable["name"]

    print(f"✅ Selected variable for distribution: {var_name} from {dataset}")

    # -------------------------------------------
    # Step 3 – Restrict filters and fields to selected dataset
    # -------------------------------------------
    filters = {
        k: v for k, v in filters.items()
        if any(var["name"] == k and var["dataset"] == dataset for var in variables)
    }
    field_names = [
        f for f in field_names
        if any(var["name"] == f and var["dataset"] == dataset for var in variables)
    ]

    # -------------------------------------------
    # Step 4 – Validate and clean filters
    # -------------------------------------------
    filters = correct_filter_values(filters, variables, dataset)
    print(f"✅ Corrected filters: {json.dumps(filters, indent=2)}")

    # -------------------------------------------
    # Step 5 – Map variable names to conceptPaths
    # -------------------------------------------
    name_to_path = {
        v["name"].lower(): v["conceptPath"]
        for v in variables if v["dataset"] == dataset
    }

    # -------------------------------------------
    # Step 6 – Build filters for API payload and generate filter description
    # -------------------------------------------
    category_filters, numeric_filters = {}, {}
    filter_desc_parts = []

    for name, condition in filters.items():
        var = next((v for v in variables if v["name"].lower() == name.lower() and v["dataset"] == dataset), None)
        if not var: continue
        path = var["conceptPath"]
        print(f"🔍 Processing filter '{name}' → {path}")

        if isinstance(condition, dict):  # numeric
            numeric_filters[path] = {k: str(v) for k, v in condition.items() if v != ""}
            min_val, max_val = condition.get("min"), condition.get("max")
            if min_val and max_val:
                filter_desc_parts.append(f"{name}_from_{min_val}_to_{max_val}")
            elif min_val:
                filter_desc_parts.append(f"{name}_gt{min_val}")
            elif max_val:
                filter_desc_parts.append(f"{name}_lt{max_val}")

        elif isinstance(condition, list):  # categorical
            category_filters[path] = condition
            filter_desc_parts.append(f"{name}_{'_'.join(v.replace(' ', '') for v in condition)}")

    filter_desc = "__".join(filter_desc_parts).replace(" ", "_").lower()

    # -------------------------------------------
    # Step 7 – Fields → conceptPaths
    # -------------------------------------------
    fields = [name_to_path[f.lower()] for f in field_names if f.lower() in name_to_path]
    if concept_path not in fields:
        fields.append(concept_path)

    # -------------------------------------------
    # Step 8 – Genomic filter
    # -------------------------------------------
    variant_filters = [{
        "categoryVariantInfoFilters": {},
        "numericVariantInfoFilters": {}
    }]

    if genomic and "gene" in genomic:
        genes = genomic["gene"]
        if genes:
            print(f"🧬 Applying genomic filter: {genes}")
            variant_filters = [{
                "categoryVariantInfoFilters": {"Gene_with_variant": genes},
                "numericVariantInfoFilters": {}
            }]
            gene_str = "_".join(genes).lower()
            filter_desc = (filter_desc or "") + f"__gene-{gene_str}"

    # -------------------------------------------
    # Step 9 – Build and send API request
    # -------------------------------------------
    query = {
        "expectedResultType": "DATAFRAME",
        "categoryFilters": category_filters,
        "numericFilters": numeric_filters,
        "fields": fields,
        "anyRecordOf": [],
        "requiredFields": [],
        "variantInfoFilters": variant_filters
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "query": query,
        "resourceUUID": "ded89b08-faa9-435c-b7c4-55b81922ee5f"
    }

    print(f"📡 Sending DISTRIBUTION query for variable: {concept_path}")
    print(json.dumps(payload, indent=2))

    response = requests.post(
        url=f"{api_url}/query/sync",
        headers=headers,
        data=json.dumps(payload)
    )
    
    print(f"🔁 Status code: {response.status_code}")

    if response.status_code != 200:
        print(f"❌ API Error: {response.text}")
        return None, None

    # -------------------------------------------
    # Step 10 – Parse CSV response and return DataFrame
    # -------------------------------------------
    try:
        df = pd.read_csv(StringIO(response.text))
        print(f"✅ Received {len(df)} rows.")

        # Select only distribution column for plotting
        distribution_col = None
        concept_path = selected_variable["conceptPath"]
        for col in df.columns:
            if col.strip().lower() == var_name.strip().lower():
                distribution_col = col
                break
            if col.strip().lower() == concept_path.strip().lower():
                distribution_col = col
                break

        if distribution_col:
            df = df[[distribution_col]]
            print(f"📊 Retained column for plotting: {distribution_col}")
        else:
            print("⚠️ Distribution variable column not found in dataframe.")
            print("🧪 Available columns:", list(df.columns))

        return df, filter_desc

    except Exception as e:
        print(f"❌ Failed to parse CSV response: {e}")
        return None, None


def plot_distribution(df: pd.DataFrame, filter_description=None):
    """
    Plots a histogram from a patient-level DataFrame.

    This function automatically identifies a numeric column that does not contain 'id' in its name,
    converts it to numeric format, bins the data into a fixed number of intervals (default 20),
    and plots the histogram with counts annotated on top of each bar.
    The resulting plot is saved as a PNG file and the filename is returned.

    Args:
        df (pd.DataFrame): A DataFrame returned from extract_distribution_dataframe(), 
                           expected to contain exactly one relevant column with numeric values.
        filter_description (str or None): Optional string to append to the filename and title 
                                          (e.g., "age_gt60_female").

    Returns:
        str or None: The filename of the saved plot, or None if plotting failed.
    """

    print(f"\nDataframe: {df.head()}\n")
    print(f"🧪 Column types:\n{df.dtypes}\n")

    # -------------------------------------------------
    # Step 1 – Identify a numeric column (not an ID)
    # -------------------------------------------------
    var_col = None
    for col in df.columns:
        if "id" not in col.lower():
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            try:
                pd.to_numeric(non_null.head(5), errors='raise')
                var_col = col
                break
            except Exception:
                continue

    if var_col is None:
        print("⚠️ No valid numeric variable found for distribution plot.")
        print("📛 Columns available:", list(df.columns))
        print("📊 First 5 rows:\n", df.head())
        return None

    # -------------------------------------------------
    # Step 2 – Clean and convert the column to numeric
    # -------------------------------------------------
    data = pd.to_numeric(df[var_col], errors='coerce').dropna()

    if data.empty:
        print(f"⚠️ Column '{var_col}' has no usable numeric values.")
        return None

    # -------------------------------------------------
    # Step 3 – Compute histogram bins and labels
    # -------------------------------------------------
    # We want a histogram with a reasonable number of slices: default 20
    num_bins = 20
    min_val, max_val = data.min(), data.max()

    # Slice edges are defined with np.linspace for regular slicing
    bins = np.linspace(min_val, max_val, num_bins + 1)

    # We count how many values fall into each bracket
    counts, edges = np.histogram(data, bins=bins)

    # Slice labels are generated for the X axis, e.g. “18-22”, “22-26”, etc.
    labels = [f"{edges[i]:.0f}-{edges[i+1]:.0f}" for i in range(len(edges) - 1)]

    # -------------------------------------------------
    # Step 4 – Plot the histogram
    # -------------------------------------------------
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, counts, color='crimson', edgecolor='black')

    # Annotate count values above each bar
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height + max(counts) * 0.01,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # -------------------------------------------------
    # Step 5 – Final formatting and export
    # -------------------------------------------------
    # Extract clean variable name from conceptPath if applicable
    clean_var_name = var_col.strip("\\").split("\\")[-1].replace("**", "").strip()

    # Formatted title with optional filter description
    if filter_description:
        filter_text = filter_description.replace("__", " - ").replace("_", " ")
        title = f"Distribution of {clean_var_name} – {filter_text}"
    else:
        title = f"Distribution of {clean_var_name}"

    plt.title(title, fontsize=14)
    plt.xlabel(var_col, fontsize=12)
    plt.ylabel("Number of Participants", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Build filename from variable and filter name
    filename_base = f"distribution_{var_col.replace(' ', '_').lower()}"
    if filter_description:
        safe_filter = filter_description.replace(" ", "")
        filename_base += f"_{safe_filter}"
    filename = f"{filename_base}.png"

    # Save to file
    plt.savefig(plot_dir / f"{filename_base}.png")
    plt.close()

    return filename

