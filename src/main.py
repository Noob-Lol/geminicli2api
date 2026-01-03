import logging
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .auth import get_credentials, get_user_project_id, onboard_user
from .gemini_routes import router as gemini_router
from .openai_routes import router as openai_router

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
    logging.info("Environment variables loaded from .env file")
except ImportError:
    logging.warning("python-dotenv not installed, .env file will not be loaded automatically")
except Exception as e:
    logging.warning(f"Could not load .env file: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create ClientSession
    app.state.client_session = aiohttp.ClientSession()

    try:
        logging.info("Starting Gemini proxy server...")

        # Check if credentials exist
        import os

        from .config import CREDENTIAL_FILE

        env_creds_json = os.getenv("GEMINI_CREDENTIALS")
        creds_file_exists = os.path.exists(CREDENTIAL_FILE)  # noqa: ASYNC240

        # Calculate auth port
        app_port = int(os.getenv("PORT", "8888"))
        auth_port = app_port + 1

        if env_creds_json or creds_file_exists:
            try:
                # Try to load existing credentials without OAuth flow first
                creds = get_credentials(allow_oauth_flow=False, auth_port=auth_port)
                if creds:
                    try:
                        proj_id = await get_user_project_id(creds, app.state.client_session)
                        if proj_id:
                            await onboard_user(creds, proj_id, app.state.client_session)
                            logging.info(f"Successfully onboarded with project ID: {proj_id}")
                        logging.info("Gemini proxy server started successfully")
                        logging.info("Authentication required - Password: see .env file")
                    except Exception as e:
                        logging.error(f"Setup failed: {str(e)}")
                        logging.warning("Server started but may not function properly until setup issues are resolved.")
                else:
                    logging.warning(
                        "Credentials file exists but could not be loaded. Server started - authentication will be required on first request.",
                    )
            except Exception as e:
                logging.error(f"Credential loading error: {str(e)}")
                logging.warning("Server started but credentials need to be set up.")
        else:
            # No credentials found - prompt user to authenticate
            logging.info("No credentials found. Starting OAuth authentication flow...")
            try:
                creds = get_credentials(allow_oauth_flow=True, auth_port=auth_port)
                if creds:
                    try:
                        proj_id = await get_user_project_id(creds, app.state.client_session)
                        if proj_id:
                            await onboard_user(creds, proj_id, app.state.client_session)
                            logging.info(f"Successfully onboarded with project ID: {proj_id}")
                        logging.info("Gemini proxy server started successfully")
                    except Exception as e:
                        logging.error(f"Setup failed: {str(e)}")
                        logging.warning("Server started but may not function properly until setup issues are resolved.")
                else:
                    logging.error(
                        "Authentication failed. Server started but will not function until credentials are provided.",
                    )
            except Exception as e:
                logging.error(f"Authentication error: {str(e)}")
                logging.warning("Server started but authentication failed.")

        logging.info("Authentication required - Password: see .env file")

    except Exception as e:
        logging.error(f"Startup error: {str(e)}")
        logging.warning("Server may not function properly.")

    yield

    # Shutdown: Close ClientSession
    if hasattr(app.state, "client_session"):
        await app.state.client_session.close()


app = FastAPI(lifespan=lifespan)

# Add CORS middleware for preflight requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


@app.options("/{full_path:path}")
async def handle_preflight(request: Request, full_path: str):
    """Handle CORS preflight requests without authentication."""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        },
    )


@app.head("/{full_path:path}")
async def handle_head(request: Request, full_path: str):
    """Handle HEAD requests without authentication."""
    return Response(status_code=200)


# Root endpoint - no authentication required
@app.get("/")
async def root():
    """
    Root endpoint providing project information.
    No authentication required.
    """
    return {
        "name": "geminicli2api",
        "description": "OpenAI-compatible API proxy for Google's Gemini models via gemini-cli",
        "purpose": "Provides both OpenAI-compatible endpoints (/v1/chat/completions) and native Gemini API endpoints for accessing Google's Gemini models",
        "version": "1.0.0",
        "endpoints": {
            "openai_compatible": {
                "chat_completions": "/v1/chat/completions",
                "models": "/v1/models",
            },
            "native_gemini": {
                "models": "/v1beta/models",
                "generate": "/v1beta/models/{model}/generateContent",
                "stream": "/v1beta/models/{model}/streamGenerateContent",
            },
            "health": "/health",
        },
        "authentication": "Required for all endpoints except root and health",
        "repository": "https://github.com/user/geminicli2api",
    }


# Health check endpoint for Docker/Hugging Face
@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "service": "geminicli2api"}


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("favicon.ico")


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def devtools_json():
    return Response(status_code=404)


app.include_router(openai_router)
app.include_router(gemini_router)
