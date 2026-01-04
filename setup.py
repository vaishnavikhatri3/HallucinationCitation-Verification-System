"""
Setup script to download required NLTK data and verify installation
"""
import nltk
import sys

def download_nltk_data():
    """Download required NLTK data"""
    print("📦 Downloading NLTK data...")
    
    data_packages = [
        'punkt',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words'
    ]
    
    for package in data_packages:
        try:
            print(f"   Checking {package}...", end=" ")
            nltk.data.find(f'tokenizers/{package}')
            print("✓ Already downloaded")
        except LookupError:
            print(f"   Downloading {package}...", end=" ")
            nltk.download(package, quiet=True)
            print("✓ Downloaded")
        except Exception as e:
            print(f"   Error: {e}")

def verify_installation():
    """Verify that required packages are installed"""
    print("\n🔍 Verifying installation...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'requests',
        'sentence_transformers',
        'transformers',
        'nltk',
        'torch',
        'numpy',
        'sklearn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages installed!")
        return True

if __name__ == "__main__":
    print("🚀 AI Hallucination Detection System - Setup\n")
    print("=" * 60)
    
    download_nltk_data()
    success = verify_installation()
    
    print("\n" + "=" * 60)
    if success:
        print("\n✅ Setup complete! You can now run:")
        print("   python main.py")
        print("   or")
        print("   python run_server.py")
    else:
        print("\n⚠️  Please install missing packages before proceeding.")
        sys.exit(1)



