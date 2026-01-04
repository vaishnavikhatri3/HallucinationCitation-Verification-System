"""
Convenience script to run the server
"""
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting AI Hallucination Detection System API...")
    print("📡 API available at http://localhost:8000")
    print("🌐 To use the web interface, run: streamlit run app.py")
    print("📚 API docs available at http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

