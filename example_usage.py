"""
Example usage of the AI Hallucination Detection System
"""
import requests
import json

# Example AI-generated text with potential hallucinations
example_text = """
According to Smith et al. (2021), GPT models reduce hallucinations by 73% [1]. 
Research shows that transformer architectures improve accuracy significantly. 
A study by Johnson (2022) found that BERT models achieve 95% accuracy in 
fact-checking tasks. However, recent work by Lee et al. (2023) contradicts 
these findings, showing only 45% accuracy. 

For more information, visit https://example-fake-url.com/research.

According to a 2024 study, AI models can now generate completely accurate 
information without any errors. This represents a breakthrough in the field.
"""

def verify_text_example():
    """Example of how to use the verification API"""
    
    print("🔍 AI Hallucination Detection System - Example Usage\n")
    print("=" * 60)
    print("\nExample Text:")
    print("-" * 60)
    print(example_text)
    print("-" * 60)
    
    # Make API request
    try:
        print("\n📡 Sending verification request...")
        response = requests.post(
            "http://localhost:8000/verify",
            json={
                "text": example_text,
                "verify_citations": True,
                "verify_facts": True
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ Verification Complete!")
            print("=" * 60)
            print(f"\n📊 Results:")
            print(f"   Overall Risk: {result['overall_risk'].upper()}")
            print(f"   Risk Score: {result['risk_score']:.1f}/100")
            print(f"\n📈 Statistics:")
            print(f"   Total Claims: {result['total_claims']}")
            print(f"   Total Citations: {result['total_citations']}")
            print(f"   Verified Claims: {result['verified_claims']}")
            print(f"   Fake Citations: {result['fake_citations']}")
            print(f"   Unverified Claims: {result['unverified_claims']}")
            print(f"   Contradicted Claims: {result['contradicted_claims']}")
            print(f"   Broken Links: {result['broken_links']}")
            
            if result['issues']:
                print(f"\n🚨 Issues Found ({len(result['issues'])}):")
                for i, issue in enumerate(result['issues'], 1):
                    print(f"\n   {i}. {issue['type'].replace('_', ' ').upper()}")
                    print(f"      Severity: {issue['severity'].upper()}")
                    print(f"      Detail: {issue['detail']}")
                    if issue.get('recommendation'):
                        print(f"      💡 {issue['recommendation']}")
            else:
                print("\n✅ No issues detected!")
            
            print("\n" + "=" * 60)
            
        else:
            print(f"\n❌ Error: HTTP {response.status_code}")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server.")
        print("   Make sure the server is running:")
        print("   python main.py")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    verify_text_example()



