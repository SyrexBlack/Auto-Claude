
import os
import sys
import uvicorn
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    from proxy.server import app
    
    port = int(os.environ.get("PROXY_PORT", 8080))
    host = os.environ.get("PROXY_HOST", "127.0.0.1")
    
    print(f"Starting RouterAI Translation Proxy on http://{host}:{port}")
    print(f"Configure Auto-Claude with ANTHROPIC_BASE_URL=http://{host}:{port}/v1")
    
    uvicorn.run(app, host=host, port=port)
