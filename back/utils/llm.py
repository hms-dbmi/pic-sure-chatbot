"""
llm.py

This module handles all interactions with Large Language Models (LLMs) via Amazon Bedrock,
as well as the formatting, validation, and correction of LLM inputs and outputs.

It includes:
- Unified LLM invocation logic (`call_bedrock_llm`)
- Schema-aligned output validation from LLMs (`validate_llm_response`)
- JSON parsing and schema validation (`robust_llm_json_parse`)
- Human-readable summaries of variables for prompts (`format_variable_context`)
- Semantic correction of filter values (`correct_filter_values`)

These tools ensure consistent communication between the user query, the LLM, and
the downstream structured filtering logic in the PIC-SURE API chatbot.

"""

import boto3  # AWS SDK for Python, used here to interact with Amazon Bedrock
import json  # For encoding/decoding JSON data
import re  # For regular expressions (e.g., pattern matching in variable names)
from difflib import get_close_matches # Import for fuzzy key matching (used to detect likely typos in metadata keys)

# ID of the LLM model to use from Amazon Bedrock (e.g., Mistral 7B Instruct)
# model_id = "mistral.mistral-7b-instruct-v0:2"
model_id = "mistral.mistral-large-2402-v1:0"
# model_id = "meta.llama3-8b-instruct-v1:0"
# model_id = "amazon.titan-text-premier-v1:0"

# Too heavy... Can use with "Provisioned throughput", but expensive
# model_id = "meta.llama3-3-70b-instruct-v1:0" # normally really good for NLP
# model_id = "meta.llama3-1-70b-instruct-v1:0"

# Create a Bedrock runtime client for the 'us-east-1' AWS region
brt = boto3.client("bedrock-runtime", region_name="us-east-1")


def call_bedrock_llm(
    prompt: str,
    model_id: str = model_id,
    max_tokens: int = 800,
    temperature: float = 0.3,
    verbose: bool = False
) -> str:
    """
    Sends a prompt to an LLM via Amazon Bedrock's 'converse' endpoint and retrieves the response.

    This function wraps all parameters needed to query a conversational LLM (e.g., Mistral),
    and returns only the text of the first message in the output. It's used by all pipeline
    components to extract metadata, filters, distributions, etc.

    Args:
        prompt (str): The raw user or system prompt to send.
        model_id (str): The specific model deployed on Bedrock (default from config).
        max_tokens (int): The maximum number of tokens to be generated in the output.
        temperature (float): Controls randomness in generation (0 = deterministic).
        verbose (bool): If True, logs the full prompt before sending (for debugging).

    Returns:
        str: The response text returned by the LLM, or an empty string if the call fails.
    """

    if verbose:
        print("\n🧠 Prompt sent to LLM:\n")
        print(prompt)
        print("\n🚀 Sending request to Bedrock...\n")

    # Construct the message for Bedrock's "converse" endpoint (chat-style prompt)
    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}]
        }
    ]

    try:
        # Send the message to the LLM with inference and performance settings
        response = brt.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig={
                "maxTokens": max_tokens,  # Limit the response length
                "temperature": temperature  # Control randomness (0 = deterministic, 1 = more random) 
            },
            performanceConfig={
                "latency": "standard"  # Balanced latency/performance tradeoff
            }
        )
        # Extract and return only the first piece of text in the response
        return response["output"]["message"]["content"][0]["text"]

    except Exception as e:
        print(f"❌ Bedrock LLM call failed: {e}")
        return ""
    

def validate_llm_response(parsed: dict, expected_schema: dict, strict_keys: bool = False) -> dict:
    """
    Validates and corrects an LLM output dictionary to conform to a specified schema.

    - Fixes capitalization and fuzzy-matches keys (e.g., "Intnet" → "intent").
    - Ensures types match the schema; replaces invalid entries with defaults.
    - Fills in missing keys with default values.
    - Removes unexpected keys if `strict_keys=True`.

    Args:
        parsed (dict): Dictionary output returned by the LLM.
        expected_schema (dict): A dict with keys and expected types (e.g., str, list, dict, None).
        strict_keys (bool): Whether to drop all keys not in the schema (default: False).

    Returns:
        dict: A validated and cleaned-up dictionary compatible with downstream processing.
    """

    corrected = {}
    normalized_keys = {k.lower(): k for k in expected_schema}
    used_keys = set()

    for key, value in parsed.items():
        key_lower = key.lower()
        correct_key = None

        # --- Match key by name or fuzzy string similarity ---
        if key_lower in normalized_keys:
            correct_key = normalized_keys[key_lower]
        else:
            close = get_close_matches(key_lower, normalized_keys.keys(), n=1, cutoff=0.8)
            if close:
                correct_key = normalized_keys[close[0]]
                print(f"🛠️ Key '{key}' corrected to '{correct_key}'")
            else:
                # Fallback: match key by value type
                for expected_key, expected_type in expected_schema.items():
                    if expected_key not in corrected:
                        if isinstance(expected_type, tuple):
                            valid_type = any(isinstance(value, t) for t in expected_type)
                        else:
                            valid_type = isinstance(value, expected_type)

                        if valid_type:
                            correct_key = expected_key
                            print(f"🧠 Fallback: key '{key}' assumed to be '{correct_key}' based on type match.")
                            break

        if not correct_key:
            if strict_keys:
                print(f"⚠️ Key '{key}' discarded (no match and strict mode)")
                continue
            else:
                correct_key = key
                print(f"⚠️ Keeping unexpected key '{key}'")

        if correct_key in corrected:
            print(f"⚠️ Skipping key '{key}' to avoid overwriting '{correct_key}'")
            continue

        # --- Validate value type or assign default ---
        expected_type = expected_schema.get(correct_key)
        if expected_type and not isinstance(value, expected_type):
            print(f"⚠️ Type mismatch for key '{correct_key}' → resetting to default.")
            value = (
                [] if expected_type == list else
                "" if expected_type == str else
                {} if expected_type == dict else
                None
            )

        corrected[correct_key] = value
        used_keys.add(correct_key)

    # --- Add missing keys with default values ---
    for k, t in expected_schema.items():
        if k not in corrected:
            print(f"⚠️ Missing key '{k}' → assigning default.")
            corrected[k] = (
                [] if t == list else
                "" if t == str else
                {} if t == dict else
                None
            )

    return corrected


def robust_llm_json_parse(response: str, schema: dict, fallback: dict, label: str = "LLM response") -> dict:
    """
    Attempts to parse and validate a JSON response from an LLM, with fallback mechanisms on failure.

    - If initial parsing fails, uses regex to clean up common issues:
        - Unclosed quotes
        - Control characters
    - If parsing still fails, returns a default fallback.

    Args:
        response (str): Raw string returned by the LLM (possibly invalid JSON).
        schema (dict): Expected schema for validation.
        fallback (dict): Default response to return if parsing fails.
        label (str): Optional label for debugging/logging purposes.

    Returns:
        dict: Parsed and validated dictionary, or fallback if unrecoverable.
    """
    try:
        parsed = json.loads(response)
        return validate_llm_response(parsed, schema)

    except json.JSONDecodeError as e:
        print(f"⚠️ {label} - JSON error: {e}")

        # Attempt 1 – Fix unclosed quotes on lines (e.g., "type": "both,)
        text_clean = re.sub(r'"(\w+)"\s*:\s*"([^"\n]*),\s*\n', r'"\1": "\2",\n', response)

        # Attempt 2 – Remove ASCII control characters (can break JSON decoding)
        text_clean = re.sub(r'[\x00-\x1F]+', ' ', text_clean)

        try:
            parsed = json.loads(text_clean)
            print(f"✅ {label} - Recovered from JSON error via regex fix.")
            return validate_llm_response(parsed, schema)

        except Exception as e2:
            print(f"❌ {label} - Still failed to parse after fix: {e2}")
            print("🔎 Raw response:\n", response)
            return fallback

    except Exception as e:
        print(f"❌ {label} - Unexpected error: {e}")
        return fallback


def format_variable_context(variables: list, max_vars: int = 150) -> str:
    """
    Builds a readable variable summary to inject into an LLM prompt.

    This function is used to inform the LLM about the available variables that
    can be used for filtering. It includes variable names, types, dataset names,
    concept paths, and value ranges when available.

    Args:
        variables (list): A list of variable dictionaries. Each dict typically contains:
                          - name (str): Variable label
                          - type (str): "categorical" or "continuous"
                          - dataset (str): Dataset name (e.g., "Synthea")
                          - conceptPath (str): Full ontology path
                          - values (list, optional): For categorical variables
                          - min (float), max (float): For continuous variables
        max_vars (int): Maximum number of variables to include (default: 150)

    Returns:
        str: Formatted summary of selected variables, suitable for LLM input.
    """

    # If no variables were provided, return a fallback message
    if not variables:
        return "No variable metadata was found.\n"

    context = ""

    # Loop over the first `max_vars` variables and construct a descriptive summary line for each
    for var in variables[:max_vars]:
        name = var.get("name", "Unknown")
        dataset = var.get("dataset", "Unknown")
        var_type = var.get("type", "Unknown")
        concept_path = var.get("conceptPath", "N/A").replace("\\", "\\\\")

        # Start the line with the main structure
        summary = f"- {name} ({var_type}, from dataset {dataset}) → {concept_path}"

        # Append values for categorical variables
        if var_type.lower() == "categorical" and var.get("values"):
            values = ", ".join(var["values"][:5])  # Show only top 5 values
            summary += f" | values: {values}"

        # Append range for continuous variables
        elif var_type.lower() == "continuous":
            min_val = var.get("min")
            max_val = var.get("max")
            summary += f" | range: {min_val} → {max_val}"

        # Add the summary line to the full context
        context += summary + "\n"

    # Add a warning if some variables were omitted due to max_vars limit
    if len(variables) > max_vars:
        context += f"\nNote: Only the first {max_vars} variables are shown out of {len(variables)} total.\n"

    return context


def correct_filter_values(filters: dict, variables: list, dataset: str) -> dict:
    """
    Ensures all filter values align with known constraints from the variable dictionary.

    - For categorical variables: keeps only values that exactly match known values.
    - For continuous variables: clips or corrects 'min' and 'max' bounds based on metadata.
    - Filters from other datasets are ignored.

    Args:
        filters (dict): Raw filters from LLM output, with potential errors or case mismatches.
        variables (list): Metadata list of all variables (name, type, values, ranges, etc.).
        dataset (str): Name of the dataset currently selected.

    Returns:
        dict: Cleaned and corrected filters, ready to be sent to the PIC-SURE API.
    """
    corrected = {}

    # Iterate over the filters suggested by the LLM
    for name, val in filters.items():
        for var in variables:
            if var.get("dataset") != dataset:
                continue
            if var["name"].lower() != name.lower():
                continue  # Case-insensitive match on variable name

            var_type = var.get("type", "").lower()
            var_name = var["name"]

            # --- Categorical filters ---
            # Correct values using those in the dictionary
            if var_type == "categorical" and isinstance(val, list):
                valid_values = var.get("values", [])
                matched = [v for v in val if v in valid_values]
                if matched:
                    corrected[var_name] = matched
                else:
                    print(f"⚠️ No valid categorical values matched for '{var_name}' → input: {val}")

            # --- Continuous filters ---
            # Remove values out of known range
            elif var_type == "continuous" and isinstance(val, dict):
                min_limit = var.get("min")
                max_limit = var.get("max")
                min_val = val.get("min")
                max_val = val.get("max")
                numeric_filter = {}

                # Handle min
                if min_val is not None:
                    try:
                        min_val_num = float(min_val)
                        if min_limit is not None and min_val_num < min_limit:
                            print(f"⚠️ 'min' value {min_val_num} for '{var_name}' is below allowed limit {min_limit}")
                            numeric_filter["min"] = str(min_limit)
                        else:
                            numeric_filter["min"] = str(min_val_num)
                    except (ValueError, TypeError):
                        print(f"⚠️ Invalid min value '{min_val}' for variable '{var_name}'")

                # Handle max
                if max_val is not None:
                    try:
                        max_val_num = float(max_val)
                        if max_limit is not None and max_val_num > max_limit:
                            print(f"⚠️ 'max' value {max_val_num} for '{var_name}' is above allowed limit {max_limit}")
                            numeric_filter["max"] = str(max_limit)
                        else:
                            numeric_filter["max"] = str(max_val_num)
                    except (ValueError, TypeError):
                        print(f"⚠️ Invalid max value '{max_val}' for variable '{var_name}'")

                if numeric_filter:
                    corrected[var_name] = numeric_filter

            break  # Skip remaining variables once matched

    return corrected
