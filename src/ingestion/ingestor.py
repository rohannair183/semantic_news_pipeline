"""
Ingestor for handling data ingestion from the Open Guardian API. This module defines the
`GuardianIngestor` class, which is responsible for fetching data from the API, and
storing it in a structured format for further use in the application.
"""

class Ingestor:
    """
    Class representing the ingestor for fetching data from the Open Guardian API. This class
    searches for articles based on topics specified in configuration and loops through each article
    to extract content. It stores the exracted content in parquet files in a specified directory.
    """
    def __init__(self, client=None):
        pass
