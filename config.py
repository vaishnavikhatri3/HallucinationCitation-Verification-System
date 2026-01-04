"""
Configuration settings for the AI Hallucination Detection System
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys (optional - some APIs work without keys)
CROSSREF_API_KEY = os.getenv("CROSSREF_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# API Endpoints
CROSSREF_API_URL = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Model Settings
NLI_MODEL = "microsoft/deberta-v3-base"  # For contradiction detection
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # For semantic similarity

# Scoring Weights
SCORE_WEIGHTS = {
    "unverified_claims": 0.4,
    "fake_citations": 0.4,
    "broken_links": 0.2
}

# Risk Thresholds
RISK_THRESHOLDS = {
    "low": 30,
    "medium": 60,
    "high": 100
}

# Citation Patterns (Regex)
CITATION_PATTERNS = {
    "apa": r"([A-Z][a-z]+(?:\s+et\s+al\.)?(?:\s+and\s+[A-Z][a-z]+)?)\s*\((\d{4})\)",
    "mla": r"([A-Z][a-z]+(?:\s+et\s+al\.)?)\s+(\d{4})",
    "ieee": r"\[(\d+)\]",
    "url": r"https?://[^\s\)]+",
    "doi": r"doi:([^\s\)]+)",
}



