"""
Convenience script to run Streamlit app
"""
import subprocess
import sys

if __name__ == "__main__":
    print("🚀 Starting Streamlit App...")
    print("📡 Make sure the API server is running: python main.py")
    print("🌐 Streamlit will open at http://localhost:8501")
    print("\nPress Ctrl+C to stop the app\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Streamlit app stopped")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)



