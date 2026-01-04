"""
Hallucination Scoring and Risk Assessment Module
Computes overall hallucination risk scores and generates user-friendly reports
"""
from typing import Dict, List
from dataclasses import dataclass, asdict
from extractors import Claim, Citation
from citation_verifier import CitationVerificationResult
from fact_verifier import FactVerificationResult
from config import SCORE_WEIGHTS, RISK_THRESHOLDS


@dataclass
class Issue:
    """Represents a detected issue"""
    type: str  # 'fake_citation', 'unverified_claim', 'contradicted_claim', 'broken_link'
    severity: str  # 'high', 'medium', 'low'
    detail: str
    location: Dict  # start_pos, end_pos
    recommendation: str = ""


@dataclass
class HallucinationReport:
    """Complete hallucination detection report"""
    overall_risk: str  # 'low', 'medium', 'high'
    risk_score: float  # 0-100
    total_claims: int
    total_citations: int
    verified_claims: int
    fake_citations: int
    unverified_claims: int
    contradicted_claims: int
    broken_links: int
    issues: List[Issue]
    detailed_results: Dict


class HallucinationScorer:
    """Computes hallucination risk scores and generates reports"""
    
    def __init__(self):
        self.weights = SCORE_WEIGHTS
        self.thresholds = RISK_THRESHOLDS
    
    def generate_report(
        self,
        claims: List[Claim],
        citations: List[Citation],
        citation_results: List[CitationVerificationResult],
        fact_results: List[FactVerificationResult],
        claim_citation_pairs: List
    ) -> HallucinationReport:
        """Generate comprehensive hallucination report"""
        
        # Count issues
        fake_citations = sum(
            1 for r in citation_results 
            if r.verification_status == "fake"
        )
        
        broken_links = sum(
            1 for r in citation_results 
            if r.citation.citation_type == "url" and not r.accessible
        )
        
        unverified_claims = sum(
            1 for r in fact_results 
            if r.verification_status in ["no_evidence", "weak"]
        )
        
        contradicted_claims = sum(
            1 for r in fact_results 
            if r.contradiction
        )
        
        verified_claims = sum(
            1 for r in fact_results 
            if r.verification_status == "supported"
        )
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(
            unverified_claims=unverified_claims,
            fake_citations=fake_citations,
            broken_links=broken_links,
            contradicted_claims=contradicted_claims,
            total_claims=len(claims)
        )
        
        # Determine overall risk level
        if risk_score <= self.thresholds["low"]:
            overall_risk = "low"
        elif risk_score <= self.thresholds["medium"]:
            overall_risk = "medium"
        else:
            overall_risk = "high"
        
        # Generate issues list
        issues = self._generate_issues(
            claims, citations, citation_results, fact_results
        )
        
        # Prepare detailed results
        detailed_results = {
            "citation_verifications": [
                {
                    "citation": r.citation.text,
                    "status": r.verification_status,
                    "exists": r.exists,
                    "accessible": r.accessible,
                    "relevance": r.relevance_score
                }
                for r in citation_results
            ],
            "fact_verifications": [
                {
                    "claim": r.claim.text,
                    "status": r.verification_status,
                    "supported": r.supported,
                    "contradicted": r.contradiction,
                    "evidence_score": r.evidence_score
                }
                for r in fact_results
            ]
        }
        
        return HallucinationReport(
            overall_risk=overall_risk,
            risk_score=risk_score,
            total_claims=len(claims),
            total_citations=len(citations),
            verified_claims=verified_claims,
            fake_citations=fake_citations,
            unverified_claims=unverified_claims,
            contradicted_claims=contradicted_claims,
            broken_links=broken_links,
            issues=issues,
            detailed_results=detailed_results
        )
    
    def _calculate_risk_score(
        self,
        unverified_claims: int,
        fake_citations: int,
        broken_links: int,
        contradicted_claims: int,
        total_claims: int
    ) -> float:
        """Calculate overall risk score (0-100)"""
        
        if total_claims == 0:
            return 0.0
        
        # Normalize counts to percentages
        unverified_ratio = (unverified_claims / total_claims) * 100
        fake_citation_ratio = (fake_citations / total_claims) * 100 if total_claims > 0 else 0
        broken_link_ratio = (broken_links / total_claims) * 100 if total_claims > 0 else 0
        contradicted_ratio = (contradicted_claims / total_claims) * 100
        
        # Weighted combination
        score = (
            unverified_ratio * self.weights["unverified_claims"] +
            fake_citation_ratio * self.weights["fake_citations"] +
            broken_link_ratio * self.weights["broken_links"] +
            contradicted_ratio * 0.3  # Contradictions are very serious
        )
        
        # Cap at 100
        return min(100.0, score)
    
    def _generate_issues(
        self,
        claims: List[Claim],
        citations: List[Citation],
        citation_results: List[CitationVerificationResult],
        fact_results: List[FactVerificationResult]
    ) -> List[Issue]:
        """Generate list of specific issues"""
        issues = []
        
        # Citation issues
        for result in citation_results:
            if result.verification_status == "fake":
                issues.append(Issue(
                    type="fake_citation",
                    severity="high",
                    detail=f"Citation '{result.citation.text}' not found in any academic database",
                    location={
                        "start": result.citation.start_pos or 0,
                        "end": result.citation.end_pos or 0
                    },
                    recommendation="Verify the citation manually or remove it if unverifiable"
                ))
            elif result.verification_status == "irrelevant":
                issues.append(Issue(
                    type="irrelevant_citation",
                    severity="medium",
                    detail=f"Citation '{result.citation.text}' exists but is not relevant to the claim",
                    location={
                        "start": result.citation.start_pos or 0,
                        "end": result.citation.end_pos or 0
                    },
                    recommendation="Find a more relevant citation or remove this one"
                ))
            elif result.citation.citation_type == "url" and not result.accessible:
                issues.append(Issue(
                    type="broken_link",
                    severity="medium",
                    detail=f"URL '{result.citation.url}' is not accessible (404 or connection error)",
                    location={
                        "start": result.citation.start_pos or 0,
                        "end": result.citation.end_pos or 0
                    },
                    recommendation="Update the URL or remove the broken link"
                ))
        
        # Fact verification issues
        for result in fact_results:
            if result.contradiction:
                issues.append(Issue(
                    type="contradicted_claim",
                    severity="high",
                    detail=f"Claim '{result.claim.text[:100]}...' contradicts available evidence",
                    location={
                        "start": result.claim.start_pos,
                        "end": result.claim.end_pos
                    },
                    recommendation="Review the claim and verify against reliable sources"
                ))
            elif result.verification_status == "no_evidence":
                issues.append(Issue(
                    type="unverified_claim",
                    severity="medium",
                    detail=f"Claim '{result.claim.text[:100]}...' has no supporting evidence found",
                    location={
                        "start": result.claim.start_pos,
                        "end": result.claim.end_pos
                    },
                    recommendation="Provide citations or verify the claim independently"
                ))
            elif result.verification_status == "weak":
                issues.append(Issue(
                    type="weak_evidence",
                    severity="low",
                    detail=f"Claim '{result.claim.text[:100]}...' has weak supporting evidence",
                    location={
                        "start": result.claim.start_pos,
                        "end": result.claim.end_pos
                    },
                    recommendation="Strengthen the claim with more reliable sources"
                ))
        
        return issues



