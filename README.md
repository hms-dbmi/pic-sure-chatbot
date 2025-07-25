# 📊 PIC-SURE Chatbot – Metadata Interaction Interface

This project provides an intelligent chatbot interface to interact with the PIC-SURE API, designed to assist researchers in exploring large-scale clinical and genomic datasets.

It operates **without Retrieval-Augmented Generation (RAG)**, relying instead on structured metadata and natural language understanding through **Large Language Models (LLMs)** powered by **Amazon Bedrock**.

Through simple, conversational prompts, users can ask:

- How many participants match a set of conditions,  
- What the distribution of a variable looks like,  
- Or what variables are available for a given research topic.

---

## 🎯 Objective

To simplify access to clinical and genomic metadata hosted in the PIC-SURE ecosystem by enabling a natural language workflow.  
This chatbot transforms unstructured research questions into structured API queries — making metadata navigation faster, more accessible, and LLM-augmented.

---

## 🧠 Key Features

- **Intent: `count`**  
  Returns the number of participants matching filters extracted from the question.

- **Intent: `distribution`**  
  Builds and visualizes a histogram for a selected continuous variable, with optional filters.

- **Intent: `information`**  
  Summarizes available datasets or variables, often grouped by relevance or concept.

- **Multi-turn conversation support**  
  Maintains user context to allow follow-up questions and refinement.

- **Metadata-only focus**  
  Uses only PIC-SURE’s metadata endpoints (e.g., `/concepts`, `/aggregate`) —  
  no direct access to patient-level data yet (HPDS-secured endpoints not included for now).

---

## 🙌 How to Use?

This section explains how to set up and run the PIC-SURE Chatbot locally.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/picsure-chatbot.git
cd picsure-chatbot/back
```

### 2. Set Up a Python Virtual Environment

We recommend using `venv` to isolate dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Requirements

All dependencies are listed in `requirements.txt`.  
Make sure this file exists at the root of `back/`.

```bash
pip install -r requirements.txt
```

### 4. Configure Access Credentials

Create or modify the `confidential.py` file (already included), and ensure it contains:

```python
PICSURE_API_URL = "https://..."
PICSURE_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."  # Secure API token
```

You can obtain your token via the PIC-SURE platform or from your institution.

### 5. Run the Chatbot

Use the pipeline entry point, to start a conversational session:

```bash
python pipeline.py
```

This will:
- Prompt the user for input
- Route the intent (count, distribution, metadata)
- Call the LLM via Amazon Bedrock
- Format and return the output

### 6. View Saved Outputs (Optional)

- Plots are saved in the `plots/` folder
- Debug info can be printed to console if enabled inside each module

### ✅ Environment Recap

- Python ≥ 3.9 recommended
- Dependencies: `boto3`, `requests`, `matplotlib`, `pandas`, `pyyaml`, etc.
- Amazon credentials must be set via AWS CLI or `~/.aws/credentials` for Bedrock access

Need help? Reach out via Issues or email the author.

---

## 🗂 Project Structure

```text
back/
├── plots/                  # Contains generated histogram images
├── prompts/                # YAML prompt templates used for LLM calls
├── utils/                  # Core logic for each chatbot intent
│   ├── count.py              # Filter extraction and count query execution
│   ├── distribution.py       # Distribution variable selection and plotting
│   ├── extract_metadata.py   # Metadata parsing (intent, search terms, dataset)
│   ├── information.py        # Natural answer generation from metadata
│   └── llm.py                # Core LLM call logic and utilities
├── context.py              # Fetches available datasets from the PIC-SURE API
├── confidential.py         # Stores PIC-SURE token and API URL
├── pipeline.py             # Main chatbot execution pipeline
```

---

## 🧱 How It Works – Chatbot Pipeline

The chatbot follows a **three-step logic**, depending on the user's intent.

```text
USER QUESTION
   │
   ▼
[Step 1] Metadata Extraction (via LLM)
   └── extract_metadata.py
       ├── intent: "count", "distribution", or "metadata"
       ├── search_terms (keywords)
       ├── dataset (if mentioned)
       └── variable type (categorical, continuous, or both)

[Step 2] Intent-specific resolution
   ├── count.py → Extracts filters + sends COUNT query to PIC-SURE API
   ├── distribution.py → Selects one variable + filters + fields → API call → DataFrame → plot
   └── information.py → Returns a natural-language summary answer (no API call)

[Step 3] Response generation
   ├── count/distribution → Structured message with number/plot
   └── information → Direct LLM answer returned as-is
```

The process is designed to be modular, so each intent type has its own logic block, input format, and output structure.

---

## 📤 Prompt Templates

All prompts sent to the LLMs are defined in a central YAML file:
```text
back/prompts/base_prompts.yml
```
Each key corresponds to a specific processing step or chatbot intent.

### 🧾 Prompt Sections

| Key name                  | Role / Intent                            |
|---------------------------|------------------------------------------|
| `metadata_extraction`     | Parses question to extract intent, dataset, and search terms |
| `count_filter_extraction` | Identifies exact filters and values from metadata |
| `distribution_extraction` | Selects a continuous variable + filters + involved fields |
| `information_response`    | Generates a natural-language summary using available variables |

### 🧩 Variable injection
`{user_question}` is used in all prompts
`{variable_context}` is used in:
- `count_filter_extraction`
- `distribution_extraction`
- `information_response`

### 💡 Prompt Design Highlights
- All prompts return strict JSON for structured parsing (except `information_response`)
- Prompts contain:
    - Domain-specific instructions (e.g., how to infer "morbid obesity")
    - Dataset coherence constraints (no mixing datasets)
    - Fallback behavior if no result is found

### 🔧 Suggested Improvements
- Let LLM return `nb_bins` in `distribution_extraction`for histogram granularity (currently fixed at 20)
- Add optional `filter_description` for labeling plots or responses
- Return flags like `"uncertainty": true` to detect edge-case answers
- Auto-inject `{datasets}` dynamically (using `context.py`)

---

## 💬 Conversational Behavior

The chatbot supports **multi-turn interactions**, which allows users to refine questions or continue exploring a topic without repeating themselves.

This modular logic ensures each intent has the right balance between memory and performance.

### `chat.previous_interaction`
Used in:
- `extract_metadata.py`
- `count.py`
- `distribution.py`

→ Only the **last exchange** is passed to keep prompts lightweight and context-specific.

### `chat.chat_history`
Used only in:
- `information.py`

→ The **entire history** is passed to the LLM, since:
- Information questions tend to be vague or exploratory
- A broader context improves relevance and reduces ambiguity

---

## 🔍 Module-by-Module Breakdown

This section describes the role of each key Python script and its logic.

### `extract_metadata.py`

```python
metadata = extract_query_metadata(
    user_question,
    previous_interaction=(previous_user, previous_bot),
    previous_extracted_metadata=self. previous_extracted_metadata
)
```

- **Role:**
Handles Step 1 of the chatbot: infers `intent`, `search_terms`, `dataset`, and optionally `type` from the user’s question.
- **LLM Prompt:** `metadata_extraction`
- **Behavior:**
  - May fall back to general exploration mode if input is vague
  - Uses `previous_interaction` as context
  - **Also passes previous extracted metadata** (search terms, etc.)  
    → This improves consistency across multi-turn queries about the same topic.

- **Output example:**
```text
{
  "intent": "count",
  "search_terms": ["sex", "gender", "age", "year"],
  "dataset": "Synthea",
  "type": null
}
```
- **Notes:**
    - `type` is currently unused, but extracted for potential filtering logic in the future.
    - Called before any API request is made
    - If previous interactions, **also passes previous extracted metadata** (search terms, etc.)  
    → This improves consistency across multi-turn queries about the same topic.


### `count.py`

```python
count_result = count_from_metadata(
  user_question,
  metadata,
  variables,
  self.token,
  self.api_url,
  self.previous_interaction
)
```

- **Role:**
Handles `intent = "count"`
    - Builds a new prompt with the question + variable context
    - Extracts filters and optional genomic conditions (genes)
    - Sends the payload to PIC-SURE `/aggregate` endpoint
- **LLM Prompt:** `count_filter_extraction`
- **Notes:**
    - Filters must all come from the same dataset
    - `correct_filter_values()` ensures exact match with categorical values

💡 **Dev Tip:**
You may improve reliability by pre-filtering variables by dataset before calling `correct_filter_values()`.

### `distribution.py`

```python
df, filter_desc = extract_distribution_dataframe(
  user_question,
  metadata,
  variables,
  self.token,
  self.api_url,
  self.previous_interaction
)

filename = plot_distribution(df, filter_description=filter_desc)
```
- **Role:**
Handles `intent = "distribution"`
    - Selects one continuous variable to plot
    - Extracts filters and genomic fields
    - Sends query → gets DataFrame → keeps only relevant column → plots histogram
- **LLM Prompt:** `distribution_extraction`
- **Plotting:**
    - Uses `matplotlib`
    - Number of bins is fixed to 20 (✅ future option to make dynamic)
- **Returns:**
    - A saved plot in `/plots/`
    - A title generated from filters

### `information.py`

```python
final_answer = information_from_metadata(
  user_question,
  variables,
  list(self.chat_history)
)
```
- **Role:**
Handles `intent = "metadata"`
    - Builds and sends prompt with variable list and user question
    - Returns raw LLM output directly (no post-processing)
- **LLM Prompt:** `information_response`
- **Behavior:**
    - Groups results by dataset
    - May mention units, types, concept paths
    - **Uses full `chat_history`**, not just previous interaction  
    → This is intentional, as "information" questions are often vague or broad. 
    Including the full context improves relevance and consistency of the LLM’s response.

## `llm.py`
- **Role:**
Contains all core logic for interacting with Amazon Bedrock.
- **Key functions:**
    - `call_bedrock_llm(prompt)` – universal LLM call
    - `robust_llm_json_parse()` – corrects common LLM formatting issues
    - `validate_llm_response()` – ensures output schema matches expectation
    - `correct_filter_values()` – maps predicted filter values to valid metadata
- **Model:**
Default is `mistral.mistral-large-2402-v1:0`, but others are available

💡 **Dev Tip:**
You could expose `model_id` and `temperature` in pipeline configs for easier tuning.

## `context.py`
- **Role:**
Auxiliary tool to retrieve the list of available datasets from the PIC-SURE API.
- **Function:** `get_available_datasets(token)`
- **Use case:**
    - To dynamically populate the `{datasets}` placeholder in `metadata_extraction`
    - Or for debugging / dataset discovery

**Note:** Not used dynamically in production yet.

---


## 🧪 Examples & Test Cases

This section outlines real user questions and how the chatbot processes them, based on your documented test cases.

### 🧠 Intent: `metadata`

**Example 1:**
> *What are the demographic variables that are available?*

The bot responds with:
```
**NHANES**
- AGE (Continuous, years) — \Nhanes\demographics\AGE
- RACE (Categorical) — \Nhanes\demographics\RACE
- Education level — \Nhanes\demographics\EDUCATION

**Synthea**
- Age (Continuous, years) — \Synthea\demographics\Age
- Race (Categorical) — \Synthea\demographics\Race
- Sex (Categorical) — \Synthea\demographics\Sex
```

**Example 2:**
> *Are there variables related to body mass index?*

Search terms inferred: `['bmi', 'body mass index', 'obesity', 'fat', 'weight']`  
Suggested variables include:
- Body Mass Index (kg/m²)
- Weight (kg)
- Total Fat (g)
- Trunk Fat (g)

---

### 🔢 Intent: `count`

**Example 1:**
> *How many women over 60 in Synthea?*

Produces:
```json
{
  "filters": {
    "AGE": { "min": "60" },
    "SEX": ["Female"]
  },
  "genomic": null
}
```
→ PIC-SURE API returns the participant count.

**Example 2:**
> *How many participants with a variant in BRCA1 and over 50 years old?*

Adds genomic filter:
```json
"genomic": {
  "gene": ["BRCA1"]
}
```

---

### 📊 Intent: `distribution`

**Example 1:**
> *What is the BMI distribution of participants with extreme obesity and age over 21?*

Returns:
```json
{
  "distribution_variable": "Body Mass Index (kg per m²)",
  "filters": {
    "Body Mass Index (kg per m²)": { "min": "40" },
    "AGE": { "min": "21" }
  },
  "fields": ["Body Mass Index (kg per m²)", "AGE"],
  "genomic": null
}
```
→ Histogram saved in `/plots/`

**Example 2:**
> what is the distribution of the age for males above 21 years old with HTR4 variant in 1000genome?

Returns:
```json
{
  "distribution_variable": "SIMULATED AGE",
  "filters": {
    "SIMULATED AGE": {"min": "21"},
    "SEX": ["male"]
  },
  "fields": [
    "SIMULATED AGE",
    "SEX"
  ],
  "genomic": {
    "gene": ["HTR4"]
  }
}
```
→ Histogram saved in `/plots/`

---

## 🛠 Developer Notes & Future Improvements

Below are areas for future work or known architectural considerations:

### 🧹 General

- Improve modularity in `pipeline.py`
- Unify logging and error handling
- Add test suite with mocked Bedrock + PIC-SURE responses

### 🧪 Prompts

- Allow LLM to return `nb_bins` for plots (currently hardcoded to 20)
- Detect and flag uncertain outputs (`"uncertainty": true`)
- Add explicit support for variable type queries (e.g., “only categorical”)

### 🔧 Filtering & Correction Logic

- Pre-filter variable list by dataset **before** running `correct_filter_values()`.

### 📦 Dataset Management

- Use `context.py` to fetch datasets dynamically and inject into prompts
- Cache variable metadata and avoid reloading in each step

---

## 🙏 Acknowledgments

This project was developed as part of a research internship at the  
**Avillach Lab, Department of Biomedical Informatics – Harvard Medical School (2025)**.
*Louis Hayot*

It integrates Amazon Bedrock for LLM calls and the PIC-SURE API for clinical/genomic data metadata access.
