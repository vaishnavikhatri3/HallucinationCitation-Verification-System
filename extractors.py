"""
Claim and Citation Extraction Module
Extracts atomic factual claims and associated citations from AI-generated text
"""
import re
import nltk
from typing import List, Dict, Tuple
from dataclasses import dataclass
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker', quiet=True)

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)


@dataclass
class Claim:
    """Represents a single factual claim"""
    text: str
    sentence: str
    start_pos: int
    end_pos: int
    confidence: float = 0.5


@dataclass
class Citation:
    """Represents a citation reference"""
    text: str
    citation_type: str  # 'apa', 'mla', 'ieee', 'url', 'doi'
    authors: List[str] = None
    year: str = None
    url: str = None
    doi: str = None
    reference_number: str = None
    start_pos: int = None
    end_pos: int = None


@dataclass
class ClaimCitationPair:
    """Pairs a claim with its associated citation"""
    claim: Claim
    citation: Citation
    proximity_score: float  # How close citation is to claim


class ClaimExtractor:
    """Extracts atomic factual claims from text"""
    
    def __init__(self):
        self.factual_indicators = [
            r'\d+%',  # Percentages
            r'\d+\.\d+',  # Decimals
            r'\d{4}',  # Years
            r'according to',
            r'research shows',
            r'studies indicate',
            r'data suggests',
            r'evidence shows',
        ]
    
    def extract_claims(self, text: str) -> List[Claim]:
        """Extract all factual claims from text"""
        sentences = sent_tokenize(text)
        claims = []
        
        for i, sentence in enumerate(sentences):
            # Check if sentence contains factual indicators
            is_factual = any(
                re.search(pattern, sentence, re.IGNORECASE)
                for pattern in self.factual_indicators
            )
            
            # Check for named entities (likely factual)
            tokens = nltk.word_tokenize(sentence)
            tagged = pos_tag(tokens)
            entities = ne_chunk(tagged)
            
            has_entities = False
            if isinstance(entities, nltk.Tree):
                has_entities = any(
                    hasattr(entity, 'label') for entity in entities
                )
            
            if is_factual or has_entities:
                # Find position in original text
                start_pos = text.find(sentence)
                end_pos = start_pos + len(sentence)
                
                claim = Claim(
                    text=sentence,
                    sentence=sentence,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    confidence=0.7 if (is_factual and has_entities) else 0.5
                )
                claims.append(claim)
        
        return claims


class CitationExtractor:
    """Extracts citations from text using pattern matching"""
    
    def __init__(self):
        from config import CITATION_PATTERNS
        self.patterns = CITATION_PATTERNS
    
    def extract_citations(self, text: str) -> List[Citation]:
        """Extract all citations from text"""
        citations = []
        
        # APA style: Author (Year)
        apa_pattern = re.compile(self.patterns["apa"], re.IGNORECASE)
        for match in apa_pattern.finditer(text):
            authors = [match.group(1).strip()]
            year = match.group(2)
            citation = Citation(
                text=match.group(0),
                citation_type="apa",
                authors=authors,
                year=year,
                start_pos=match.start(),
                end_pos=match.end()
            )
            citations.append(citation)
        
        # MLA style
        mla_pattern = re.compile(self.patterns["mla"], re.IGNORECASE)
        for match in mla_pattern.finditer(text):
            authors = [match.group(1).strip()]
            year = match.group(2)
            citation = Citation(
                text=match.group(0),
                citation_type="mla",
                authors=authors,
                year=year,
                start_pos=match.start(),
                end_pos=match.end()
            )
            citations.append(citation)
        
        # IEEE style: [1], [2], etc.
        ieee_pattern = re.compile(self.patterns["ieee"])
        for match in ieee_pattern.finditer(text):
            citation = Citation(
                text=match.group(0),
                citation_type="ieee",
                reference_number=match.group(1),
                start_pos=match.start(),
                end_pos=match.end()
            )
            citations.append(citation)
        
        # URLs
        url_pattern = re.compile(self.patterns["url"])
        for match in url_pattern.finditer(text):
            citation = Citation(
                text=match.group(0),
                citation_type="url",
                url=match.group(0),
                start_pos=match.start(),
                end_pos=match.end()
            )
            citations.append(citation)
        
        # DOIs
        doi_pattern = re.compile(self.patterns["doi"], re.IGNORECASE)
        for match in doi_pattern.finditer(text):
            citation = Citation(
                text=match.group(0),
                citation_type="doi",
                doi=match.group(1),
                start_pos=match.start(),
                end_pos=match.end()
            )
            citations.append(citation)
        
        return citations
    
    def pair_claims_with_citations(
        self, 
        claims: List[Claim], 
        citations: List[Citation]
    ) -> List[ClaimCitationPair]:
        """Pair claims with their nearest citations"""
        pairs = []
        
        for claim in claims:
            # Find closest citation
            closest_citation = None
            min_distance = float('inf')
            
            for citation in citations:
                # Calculate proximity (character distance)
                if citation.start_pos is not None and claim.start_pos is not None:
                    distance = abs(citation.start_pos - claim.end_pos)
                    if distance < min_distance and distance < 200:  # Within 200 chars
                        min_distance = distance
                        closest_citation = citation
            
            if closest_citation:
                proximity_score = 1.0 / (1.0 + min_distance / 100.0)
                pairs.append(ClaimCitationPair(
                    claim=claim,
                    citation=closest_citation,
                    proximity_score=proximity_score
                ))
            else:
                # Claim without citation
                pairs.append(ClaimCitationPair(
                    claim=claim,
                    citation=None,
                    proximity_score=0.0
                ))
        
        return pairs


class TextAnalyzer:
    """Main analyzer that combines claim and citation extraction"""
    
    def __init__(self):
        self.claim_extractor = ClaimExtractor()
        self.citation_extractor = CitationExtractor()
    
    def analyze(self, text: str) -> Dict:
        """Analyze text and extract claims and citations"""
        claims = self.claim_extractor.extract_claims(text)
        citations = self.citation_extractor.extract_citations(text)
        pairs = self.citation_extractor.pair_claims_with_citations(claims, citations)
        
        return {
            "claims": claims,
            "citations": citations,
            "pairs": pairs,
            "total_claims": len(claims),
            "total_citations": len(citations),
            "claims_with_citations": sum(1 for p in pairs if p.citation is not None),
            "claims_without_citations": sum(1 for p in pairs if p.citation is None)
        }

