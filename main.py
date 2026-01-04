"""
Main FastAPI Application
Entry point for the AI Hallucination Detection System
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from extractors import TextAnalyzer
from citation_verifier import BatchCitationVerifier
from fact_verifier import BatchFactVerifier
from scorer import HallucinationScorer

app = FastAPI(
    title="AI Hallucination Detection System",
    description="Detects and verifies factual claims and citations in AI-generated content",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
text_analyzer = TextAnalyzer()
citation_verifier = BatchCitationVerifier()
fact_verifier = BatchFactVerifier()
scorer = HallucinationScorer()


class VerificationRequest(BaseModel):
    """Request model for text verification"""
    text: str
    verify_citations: bool = True
    verify_facts: bool = True


class VerificationResponse(BaseModel):
    """Response model for verification results"""
    overall_risk: str
    risk_score: float
    total_claims: int
    total_citations: int
    verified_claims: int
    fake_citations: int
    unverified_claims: int
    contradicted_claims: int
    broken_links: int
    issues: List[dict]
    detailed_results: dict


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Hallucination Detection System API",
        "version": "1.0.0",
        "endpoints": {
            "/verify": "POST - Verify AI-generated text",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/verify", response_model=VerificationResponse)
async def verify_text(request: VerificationRequest):
    """
    Main verification endpoint
    
    Analyzes AI-generated text and:
    1. Extracts claims and citations
    2. Verifies citations exist and are accessible
    3. Verifies factual claims against trusted sources
    4. Generates hallucination risk score and report
    """
    try:
        # Step 1: Extract claims and citations
        analysis = text_analyzer.analyze(request.text)
        claims = analysis["claims"]
        citations = analysis["citations"]
        pairs = analysis["pairs"]
        
        # Step 2: Verify citations (if enabled)
        citation_results = []
        if request.verify_citations and citations:
            # Create claim text mapping for relevance
            claim_texts = {
                pair.citation.text: pair.claim.text
                for pair in pairs
                if pair.citation
            }
            citation_results = citation_verifier.verify_citations(
                citations, 
                claim_texts
            )
        else:
            # Create empty results for unverified citations
            from citation_verifier import CitationVerificationResult
            citation_results = [
                CitationVerificationResult(
                    citation=c,
                    exists=False,
                    accessible=False,
                    relevance_score=0.0,
                    verification_status="unknown",
                    details={"skipped": "Citation verification disabled"}
                )
                for c in citations
            ]
        
        # Step 3: Verify facts (if enabled)
        fact_results = []
        if request.verify_facts and claims:
            fact_results = fact_verifier.verify_claims(claims)
        else:
            # Create empty results for unverified claims
            from fact_verifier import FactVerificationResult
            fact_results = [
                FactVerificationResult(
                    claim=c,
                    supported=False,
                    contradiction=False,
                    evidence_score=0.0,
                    verification_status="unknown",
                    evidence_sources=[],
                    contradiction_details="Fact verification disabled"
                )
                for c in claims
            ]
        
        # Step 4: Generate report
        report = scorer.generate_report(
            claims=claims,
            citations=citations,
            citation_results=citation_results,
            fact_results=fact_results,
            claim_citation_pairs=pairs
        )
        
        # Convert to response format
        return VerificationResponse(
            overall_risk=report.overall_risk,
            risk_score=report.risk_score,
            total_claims=report.total_claims,
            total_citations=report.total_citations,
            verified_claims=report.verified_claims,
            fake_citations=report.fake_citations,
            unverified_claims=report.unverified_claims,
            contradicted_claims=report.contradicted_claims,
            broken_links=report.broken_links,
            issues=[
                {
                    "type": issue.type,
                    "severity": issue.severity,
                    "detail": issue.detail,
                    "location": issue.location,
                    "recommendation": issue.recommendation
                }
                for issue in report.issues
            ],
            detailed_results=report.detailed_results
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



