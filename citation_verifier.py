"""
Citation Verification Module
Verifies if citations exist, are accessible, and match the claims
"""
import requests
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from extractors import Citation
import time

from config import (
    CROSSREF_API_URL,
    SEMANTIC_SCHOLAR_API_URL,
    CROSSREF_API_KEY,
    SEMANTIC_SCHOLAR_API_KEY
)


@dataclass
class CitationVerificationResult:
    """Result of citation verification"""
    citation: Citation
    exists: bool
    accessible: bool
    relevance_score: float  # 0-1, how relevant to the claim
    verification_status: str  # 'verified', 'fake', 'irrelevant', 'unknown'
    details: Dict
    matched_paper: Optional[Dict] = None


class CitationVerifier:
    """Verifies citations using multiple academic databases"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AI-Hallucination-Checker/1.0'
        })
    
    def verify_citation(self, citation: Citation, claim_text: str = "") -> CitationVerificationResult:
        """Verify a single citation"""
        # Try different verification methods based on citation type
        if citation.citation_type == "doi":
            return self._verify_doi(citation, claim_text)
        elif citation.citation_type == "url":
            return self._verify_url(citation, claim_text)
        elif citation.citation_type in ["apa", "mla"]:
            return self._verify_author_year(citation, claim_text)
        elif citation.citation_type == "ieee":
            return self._verify_ieee(citation, claim_text)
        else:
            return CitationVerificationResult(
                citation=citation,
                exists=False,
                accessible=False,
                relevance_score=0.0,
                verification_status="unknown",
                details={"error": "Unknown citation type"}
            )
    
    def _verify_doi(self, citation: Citation, claim_text: str) -> CitationVerificationResult:
        """Verify DOI citation"""
        doi = citation.doi or citation.text.replace("doi:", "").strip()
        
        # Try CrossRef
        try:
            response = self.session.get(
                f"{CROSSREF_API_URL}/{doi}",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    paper = data.get("message", {})
                    relevance = self._calculate_relevance(paper, claim_text)
                    
                    return CitationVerificationResult(
                        citation=citation,
                        exists=True,
                        accessible=True,
                        relevance_score=relevance,
                        verification_status="verified" if relevance > 0.5 else "irrelevant",
                        details={"source": "crossref", "paper": paper},
                        matched_paper=paper
                    )
        except Exception as e:
            pass
        
        return CitationVerificationResult(
            citation=citation,
            exists=False,
            accessible=False,
            relevance_score=0.0,
            verification_status="fake",
            details={"error": "DOI not found in CrossRef"}
        )
    
    def _verify_url(self, citation: Citation, claim_text: str) -> CitationVerificationResult:
        """Verify URL citation"""
        url = citation.url or citation.text
        
        try:
            response = self.session.head(url, timeout=10, allow_redirects=True)
            accessible = response.status_code == 200
            
            if accessible:
                # Try to get more info about the page
                try:
                    full_response = self.session.get(url, timeout=10)
                    # Simple relevance check based on claim keywords
                    relevance = self._simple_text_relevance(full_response.text[:1000], claim_text)
                except:
                    relevance = 0.5  # Default if we can't analyze content
                
                return CitationVerificationResult(
                    citation=citation,
                    exists=True,
                    accessible=True,
                    relevance_score=relevance,
                    verification_status="verified" if relevance > 0.3 else "irrelevant",
                    details={"status_code": response.status_code, "url": url}
                )
            else:
                return CitationVerificationResult(
                    citation=citation,
                    exists=True,
                    accessible=False,
                    relevance_score=0.0,
                    verification_status="fake",
                    details={"status_code": response.status_code, "error": "URL not accessible"}
                )
        except Exception as e:
            return CitationVerificationResult(
                citation=citation,
                exists=False,
                accessible=False,
                relevance_score=0.0,
                verification_status="fake",
                details={"error": str(e)}
            )
    
    def _verify_author_year(self, citation: Citation, claim_text: str) -> CitationVerificationResult:
        """Verify author-year citation (APA/MLA style)"""
        authors = citation.authors or []
        year = citation.year
        
        if not authors or not year:
            return CitationVerificationResult(
                citation=citation,
                exists=False,
                accessible=False,
                relevance_score=0.0,
                verification_status="unknown",
                details={"error": "Missing author or year"}
            )
        
        # Try Semantic Scholar API
        try:
            # Search by author and year
            query = f"{authors[0]} {year}"
            params = {
                "query": query,
                "limit": 5,
                "fields": "title,authors,year,abstract,url"
            }
            
            if SEMANTIC_SCHOLAR_API_KEY:
                self.session.headers.update({
                    'x-api-key': SEMANTIC_SCHOLAR_API_KEY
                })
            
            response = self.session.get(
                SEMANTIC_SCHOLAR_API_URL,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                papers = data.get("data", [])
                
                # Find best match
                best_match = None
                best_relevance = 0.0
                
                for paper in papers:
                    # Check if year matches
                    if str(paper.get("year")) == str(year):
                        # Check if author matches
                        paper_authors = [a.get("name", "") for a in paper.get("authors", [])]
                        author_match = any(
                            any(auth.lower() in pa.lower() for auth in authors)
                            for pa in paper_authors
                        )
                        
                        if author_match:
                            relevance = self._calculate_relevance(paper, claim_text)
                            if relevance > best_relevance:
                                best_relevance = relevance
                                best_match = paper
                
                if best_match:
                    return CitationVerificationResult(
                        citation=citation,
                        exists=True,
                        accessible=True,
                        relevance_score=best_relevance,
                        verification_status="verified" if best_relevance > 0.5 else "irrelevant",
                        details={"source": "semantic_scholar", "paper": best_match},
                        matched_paper=best_match
                    )
        except Exception as e:
            pass
        
        # Try CrossRef as fallback
        try:
            query = f"{authors[0]} {year}"
            params = {"query": query, "rows": 5}
            response = self.session.get(
                CROSSREF_API_URL,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("message", {}).get("items", [])
                
                for item in items:
                    item_year = item.get("published-print", {}).get("date-parts", [[None]])[0][0]
                    if str(item_year) == str(year):
                        relevance = self._calculate_relevance(item, claim_text)
                        if relevance > 0.3:
                            return CitationVerificationResult(
                                citation=citation,
                                exists=True,
                                accessible=True,
                                relevance_score=relevance,
                                verification_status="verified" if relevance > 0.5 else "irrelevant",
                                details={"source": "crossref", "paper": item},
                                matched_paper=item
                            )
        except Exception as e:
            pass
        
        return CitationVerificationResult(
            citation=citation,
            exists=False,
            accessible=False,
            relevance_score=0.0,
            verification_status="fake",
            details={"error": "Citation not found in academic databases"}
        )
    
    def _verify_ieee(self, citation: Citation, claim_text: str) -> CitationVerificationResult:
        """Verify IEEE-style numbered citation"""
        # IEEE citations are harder to verify without full reference list
        # Mark as unknown for now
        return CitationVerificationResult(
            citation=citation,
            exists=False,
            accessible=False,
            relevance_score=0.0,
            verification_status="unknown",
            details={"error": "IEEE citations require full reference list for verification"}
        )
    
    def _calculate_relevance(self, paper: Dict, claim_text: str) -> float:
        """Calculate how relevant a paper is to the claim"""
        if not claim_text:
            return 0.5
        
        # Extract keywords from claim
        claim_words = set(re.findall(r'\b\w+\b', claim_text.lower()))
        
        # Get paper text (title, abstract)
        paper_text = ""
        if isinstance(paper, dict):
            paper_text += paper.get("title", "").lower()
            paper_text += " " + paper.get("abstract", "").lower()
        
        if not paper_text:
            return 0.3
        
        # Simple keyword overlap
        paper_words = set(re.findall(r'\b\w+\b', paper_text))
        overlap = len(claim_words & paper_words)
        total = len(claim_words)
        
        if total == 0:
            return 0.5
        
        relevance = min(1.0, overlap / total)
        return relevance
    
    def _simple_text_relevance(self, text: str, claim_text: str) -> float:
        """Simple text-based relevance calculation"""
        claim_words = set(re.findall(r'\b\w{4,}\b', claim_text.lower()))
        text_words = set(re.findall(r'\b\w{4,}\b', text.lower()))
        
        if not claim_words:
            return 0.5
        
        overlap = len(claim_words & text_words)
        return min(1.0, overlap / len(claim_words))


class BatchCitationVerifier:
    """Verifies multiple citations with rate limiting"""
    
    def __init__(self):
        self.verifier = CitationVerifier()
        self.delay = 0.5  # Delay between requests to respect rate limits
    
    def verify_citations(
        self, 
        citations: List[Citation], 
        claim_texts: Dict[str, str] = None
    ) -> List[CitationVerificationResult]:
        """Verify multiple citations"""
        results = []
        claim_texts = claim_texts or {}
        
        for citation in citations:
            claim_text = claim_texts.get(citation.text, "")
            result = self.verifier.verify_citation(citation, claim_text)
            results.append(result)
            time.sleep(self.delay)  # Rate limiting
        
        return results



