import os
from contextlib import asynccontextmanager

import aiohttp
from anyio import Path
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .auth import get_credentials, get_user_project_id, onboard_user
from .config import CREDENTIAL_FILE
from .gemini_routes import router as gemini_router
from .openai_routes import router as openai_router
from .utils import logger

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
    logger.info("Environment variables loaded from .env file")
except ImportError:
    logger.warning("python-dotenv not installed, .env file will not be loaded automatically")
except Exception as e:
    logger.warning(f"Could not load .env file: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create ClientSession
    app.state.client_session = aiohttp.ClientSession()

    logger.info("Starting Gemini proxy server...")

    # Check if credentials exist
    env_creds_json = os.getenv("GEMINI_CREDENTIALS")
    creds_file_exists = await Path(CREDENTIAL_FILE).exists()

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
                        logger.info("Successfully onboarded with project ID: %s", proj_id)
                    logger.info("Gemini proxy server started successfully")
                    logger.info("Authentication required - Password: see .env file")
                except Exception as e:
                    logger.error(f"Setup failed: {e!s}")
                    logger.warning("Server started but may not function properly until setup issues are resolved.")
            else:
                logger.warning(
                    "Credentials file exists but could not be loaded. Server started - authentication will be required on first request.",
                )
        except Exception as e:
            logger.error(f"Credential loading error: {e!s}")
            logger.warning("Server started but credentials need to be set up.")
    else:
        # No credentials found - prompt user to authenticate
        logger.info("No credentials found. Starting OAuth authentication flow...")
        try:
            creds = get_credentials(allow_oauth_flow=True, auth_port=auth_port)
            if creds:
                try:
                    proj_id = await get_user_project_id(creds, app.state.client_session)
                    if proj_id:
                        await onboard_user(creds, proj_id, app.state.client_session)
                        logger.info("Successfully onboarded with project ID: %s", proj_id)
                    logger.info("Gemini proxy server started successfully")
                except Exception as e:
                    logger.error(f"Setup failed: {e!s}")
                    logger.warning("Server started but may not function properly until setup issues are resolved.")
            else:
                logger.error(
                    "Authentication failed. Server started but will not function until credentials are provided.",
                )
        except Exception as e:
            logger.error(f"Authentication error: {e!s}")
            logger.warning("Server started but authentication failed.")

    logger.info("Authentication required - Password: see .env file")

    yield

    # Shutdown: Close ClientSession
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


@app.head("/{full_path:path}")
async def handle_head(request: Request, full_path: str):
    """Handle HEAD requests without authentication."""
    return Response("OK", headers={"X-Status": "OK"})


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
