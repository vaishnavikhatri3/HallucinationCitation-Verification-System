"""
Fact Verification Module
Verifies factual claims using retrieval and Natural Language Inference (NLI)
"""
import re
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from extractors import Claim
from config import NLI_MODEL, EMBEDDING_MODEL


@dataclass
class FactVerificationResult:
    """Result of fact verification"""
    claim: Claim
    supported: bool
    contradiction: bool
    evidence_score: float  # 0-1, strength of evidence
    verification_status: str  # 'supported', 'contradicted', 'weak', 'no_evidence'
    evidence_sources: List[Dict]
    contradiction_details: Optional[str] = None


class FactVerifier:
    """Verifies factual claims using multiple strategies"""
    
    def __init__(self):
        # Load embedding model for semantic similarity
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Load NLI model for contradiction detection
        try:
            self.nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
            self.nli_model.eval()
        except Exception as e:
            print(f"Warning: Could not load NLI model: {e}")
            self.nli_model = None
            self.nli_tokenizer = None
    
    def verify_claim(self, claim: Claim) -> FactVerificationResult:
        """Verify a single factual claim"""
        # Step 1: Retrieve evidence from trusted sources
        evidence_sources = self._retrieve_evidence(claim.text)
        
        if not evidence_sources:
            return FactVerificationResult(
                claim=claim,
                supported=False,
                contradiction=False,
                evidence_score=0.0,
                verification_status="no_evidence",
                evidence_sources=[]
            )
        
        # Step 2: Check for contradictions using NLI
        contradiction_result = self._check_contradiction(claim.text, evidence_sources)
        
        # Step 3: Calculate evidence score
        evidence_score = self._calculate_evidence_score(claim.text, evidence_sources)
        
        # Determine verification status
        if contradiction_result["has_contradiction"]:
            verification_status = "contradicted"
        elif evidence_score > 0.7:
            verification_status = "supported"
        elif evidence_score > 0.4:
            verification_status = "weak"
        else:
            verification_status = "no_evidence"
        
        return FactVerificationResult(
            claim=claim,
            supported=evidence_score > 0.5 and not contradiction_result["has_contradiction"],
            contradiction=contradiction_result["has_contradiction"],
            evidence_score=evidence_score,
            verification_status=verification_status,
            evidence_sources=evidence_sources,
            contradiction_details=contradiction_result.get("details")
        )
    
    def _retrieve_evidence(self, claim_text: str) -> List[Dict]:
        """Retrieve evidence from trusted sources"""
        evidence_sources = []
        
        # Extract key terms for search
        search_query = self._extract_search_query(claim_text)
        
        # Search Wikipedia
        wiki_evidence = self._search_wikipedia(search_query)
        if wiki_evidence:
            evidence_sources.extend(wiki_evidence)
        
        # Search Google Scholar (via API or scraping)
        scholar_evidence = self._search_scholar(search_query)
        if scholar_evidence:
            evidence_sources.extend(scholar_evidence)
        
        return evidence_sources[:5]  # Limit to top 5 sources
    
    def _extract_search_query(self, claim_text: str) -> str:
        """Extract search query from claim"""
        # Remove common phrases
        claim_text = re.sub(r'\baccording to\b', '', claim_text, flags=re.IGNORECASE)
        claim_text = re.sub(r'\bresearch shows\b', '', claim_text, flags=re.IGNORECASE)
        claim_text = re.sub(r'\bstudies indicate\b', '', claim_text, flags=re.IGNORECASE)
        
        # Extract key phrases (nouns and numbers)
        words = claim_text.split()
        # Keep important words (length > 3, or numbers)
        key_words = [
            w for w in words 
            if len(w) > 3 or re.match(r'\d+', w)
        ][:10]  # Limit to 10 words
        
        return " ".join(key_words)
    
    def _search_wikipedia(self, query: str) -> List[Dict]:
        """Search Wikipedia for evidence"""
        try:
            # Use Wikipedia API
            api_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")
            response = requests.get(api_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return [{
                    "source": "wikipedia",
                    "title": data.get("title", ""),
                    "text": data.get("extract", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
                }]
        except Exception as e:
            pass
        
        # Fallback: search API
        try:
            search_url = "https://en.wikipedia.org/api/rest_v1/page/search"
            params = {"q": query, "limit": 3}
            response = requests.get(search_url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get("pages", [])[:3]:
                    results.append({
                        "source": "wikipedia",
                        "title": item.get("title", ""),
                        "text": item.get("extract", ""),
                        "url": f"https://en.wikipedia.org/wiki/{item.get('key', '')}"
                    })
                return results
        except Exception as e:
            pass
        
        return []
    
    def _search_scholar(self, query: str) -> List[Dict]:
        """Search Google Scholar (simplified - would need proper API in production)"""
        # In production, use Semantic Scholar API or Google Scholar API
        # For now, return empty (can be extended)
        return []
    
    def _check_contradiction(
        self, 
        claim_text: str, 
        evidence_sources: List[Dict]
    ) -> Dict:
        """Check if evidence contradicts the claim using NLI"""
        if not self.nli_model or not evidence_sources:
            return {"has_contradiction": False, "details": None}
        
        try:
            contradictions = []
            
            for source in evidence_sources:
                evidence_text = source.get("text", "")
                if not evidence_text:
                    continue
                
                # Truncate to model's max length
                max_length = 512
                claim_truncated = claim_text[:max_length//2]
                evidence_truncated = evidence_text[:max_length//2]
                
                # Tokenize
                inputs = self.nli_tokenizer(
                    claim_truncated,
                    evidence_truncated,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length
                )
                
                # Get prediction
                with torch.no_grad():
                    outputs = self.nli_model(**inputs)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Check for contradiction (assuming 3-class: entailment, neutral, contradiction)
                # Model output format may vary - DeBERTa models typically have 3 classes
                if probs.shape[1] >= 3:
                    # Index 0: entailment, 1: neutral, 2: contradiction
                    contradiction_prob = probs[0][2].item() if probs.shape[1] > 2 else 0.0
                    if contradiction_prob > 0.5:
                        contradictions.append({
                            "source": source.get("title", "Unknown"),
                            "probability": contradiction_prob
                        })
                elif probs.shape[1] == 2:
                    # Binary classification - check if second class is contradiction
                    contradiction_prob = probs[0][1].item()
                    if contradiction_prob > 0.5:
                        contradictions.append({
                            "source": source.get("title", "Unknown"),
                            "probability": contradiction_prob
                        })
            
            if contradictions:
                return {
                    "has_contradiction": True,
                    "details": f"Contradiction found in {len(contradictions)} source(s)"
                }
        except Exception as e:
            # If NLI fails, fall back to semantic similarity
            pass
        
        return {"has_contradiction": False, "details": None}
    
    def _calculate_evidence_score(
        self, 
        claim_text: str, 
        evidence_sources: List[Dict]
    ) -> float:
        """Calculate how well evidence supports the claim"""
        if not evidence_sources:
            return 0.0
        
        # Use semantic similarity
        claim_embedding = self.embedding_model.encode([claim_text])[0]
        
        similarities = []
        for source in evidence_sources:
            evidence_text = source.get("text", "")
            if evidence_text:
                evidence_embedding = self.embedding_model.encode([evidence_text])[0]
                similarity = cosine_similarity(
                    [claim_embedding],
                    [evidence_embedding]
                )[0][0]
                similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        # Average similarity, weighted by source quality
        avg_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)
        
        # Combine average and max (max shows best match, avg shows consistency)
        score = (avg_similarity * 0.6 + max_similarity * 0.4)
        
        return float(score)


class BatchFactVerifier:
    """Verifies multiple claims efficiently"""
    
    def __init__(self):
        self.verifier = FactVerifier()
    
    def verify_claims(self, claims: List[Claim]) -> List[FactVerificationResult]:
        """Verify multiple claims"""
        results = []
        for claim in claims:
            result = self.verifier.verify_claim(claim)
            results.append(result)
        return results

