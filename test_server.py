from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    logger.info("Received request to root endpoint")
    return """
    <html>
        <head>
            <title>Test Server</title>
            <style>
                body { 
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                h1 { color: #2c3e50; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Hello World!</h1>
                <p>The server is running successfully!</p>
                <p>Try visiting the <a href="/health">health check endpoint</a>.</p>
            </div>
        </body>
    </html>
    """

@app.get("/health", response_class=HTMLResponse)
async def health():
    logger.info("Received request to health endpoint")
    return """
    <html>
        <head>
            <title>Health Check</title>
            <style>
                body { 
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                h1 { color: #27ae60; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>System Status: Healthy</h1>
                <p>All systems are operational!</p>
                <p><a href="/">Back to home</a></p>
            </div>
        </body>
    </html>
    """

if __name__ == "__main__":
    logger.info("Starting server on port 8080...")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info") 