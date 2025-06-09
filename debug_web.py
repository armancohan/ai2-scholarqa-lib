#!/usr/bin/env python3

import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return HTMLResponse("<html><body><h1>Debug Test</h1></body></html>")

if __name__ == "__main__":
    import uvicorn
    print("Starting debug server...")
    uvicorn.run(app, host="0.0.0.0", port=8003)