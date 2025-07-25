# Don't call this document "secret.py", it will create a bug in the numpy library importing process.

PICSURE_API_URL = "https://nhanes.hms.harvard.edu/picsure"
PICSURE_TOKEN = "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJMT05HX1RFUk1fVE9LRU58Z29vZ2xlLW9hdXRoMnwxMTE0OTI3MDU1MTQ1NDk3NDMwMDciLCJuYW1lIjoiZ29vZ2xlLW9hdXRoMnwxMTE0OTI3MDU1MTQ1NDk3NDMwMDciLCJlbWFpbCI6ImxvdWlzaGF5b3Q1QGdtYWlsLmNvbSIsImp0aSI6IndoYXRldmVyIiwiaWF0IjoxNzUwODY5NzI4LCJpc3MiOiJlZHUuaGFydmFyZC5obXMuZGJtaS5wc2FtYSIsImV4cCI6MTc1MzQ2MTcyOH0.eq1vJ18j8MWpv5TQUK-htGmJwIXm9f1gH4ieUBciFuI3g-QBoqRW6eluZ-Mfv8yNeiypcuTx5IB0lp1mQW93mA"

def get_picsure_token() -> str:
    """
    Returns the PICSURE token.
    """
    return PICSURE_TOKEN
