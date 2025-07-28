"""
Template of confidential.py, which has to be in .gitignore for security reasons.
"""

# Don't call this document "secret.py", it will create a bug in the numpy library importing process.

PICSURE_API_URL = "https://nhanes.hms.harvard.edu/picsure"
PICSURE_TOKEN = "REPLACE_ME"

def get_picsure_token() -> str:
    """
    Returns the PICSURE token.
    """
    return PICSURE_TOKEN
