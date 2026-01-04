# Quick Start Guide

Get up and running with the AI Hallucination Detection System in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Run Setup

```bash
python setup.py
```

This will:
- Download required NLTK data
- Verify all packages are installed

## Step 3: Start the Servers

You need to run two servers:

**Terminal 1 - API Server:**
```bash
python main.py
```

**Terminal 2 - Streamlit App:**
```bash
streamlit run app.py
```

Or use the convenience script:
```bash
python run_streamlit.py
```

The API will be available at `http://localhost:8000`  
The Streamlit app will open at `http://localhost:8501`

## Step 4: Use the System

### Option A: Streamlit Web Interface (Recommended)

1. **Start the API server** (Terminal 1):
   ```bash
   python main.py
   ```

2. **Start Streamlit** (Terminal 2):
   ```bash
   streamlit run app.py
   ```

3. The browser will open automatically. Paste your text and click "Verify Text"!

### Option B: API Directly

```bash
curl -X POST "http://localhost:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "According to Smith et al. (2021), GPT models reduce hallucinations by 73%.",
    "verify_citations": true,
    "verify_facts": true
  }'
```

### Option C: Python Script

```bash
python example_usage.py
```

## Understanding the Results

- **Risk Level**: Overall assessment (Low/Medium/High)
- **Risk Score**: Numerical score 0-100
- **Issues**: Specific problems found with recommendations
- **Statistics**: Breakdown of claims, citations, and verifications

## Troubleshooting

### "Connection refused" error
- Make sure the server is running: `python main.py`
- Check that port 8000 is not in use

### "Module not found" error
- Run: `pip install -r requirements.txt`
- Verify with: `python setup.py`

### Slow verification
- First run downloads ML models (can take a few minutes)
- Citation verification makes API calls (rate-limited)
- Consider disabling citation verification for faster results

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Customize settings in `config.py`
- Add your API keys to `.env` for better citation verification

---

**Happy fact-checking! 🔍**

