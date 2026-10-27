#!/usr/bin/env python3
"""Run the FastAPI Fraud Risk Engine web service and dashboard."""
import argparse
import sys
from pathlib import Path
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Start the Fraud Risk Engine API service.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change.")
    args = parser.parse_args()

    print(f"[*] Starting Fraud Risk Engine on http://{args.host}:{args.port}")
    uvicorn.run("fraud_engine.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
