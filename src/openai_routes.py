"""
OpenAI API Routes - Handles OpenAI-compatible endpoints.
This module provides OpenAI-compatible endpoints that transform requests/responses
and delegate to the Google API client.
"""

import asyncio
import json
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from .auth import authenticate_user
from .config import SUPPORTED_MODELS
from .google_api_client import build_gemini_payload_from_openai, send_gemini_request
from .models import OpenAIChatCompletionRequest
from .openai_transformers import (
    gemini_response_to_openai,
    gemini_stream_chunk_to_openai,
    openai_request_to_gemini,
)
from .utils import logger

router = APIRouter()


@router.post("/v1/chat/completions")
async def openai_chat_completions(
    request: OpenAIChatCompletionRequest,
    http_request: Request,
    username: Annotated[str, Depends(authenticate_user)],
):
    """
    OpenAI-compatible chat completions endpoint.
    Transforms OpenAI requests to Gemini format, sends to Google API,
    and transforms responses back to OpenAI format.
    """
    try:
        logger.info(f"OpenAI chat completion request: model={request.model}, stream={request.stream}")

        # Transform OpenAI request to Gemini format
        gemini_request_data = openai_request_to_gemini(request)

        # Build the payload for Google API
        gemini_payload = build_gemini_payload_from_openai(gemini_request_data)

    except Exception as e:
        logger.error(f"Error processing OpenAI request: {e!s}")
        return _create_error_response(f"Request processing failed: {e!s}", 400, "invalid_request_error")

    if request.stream:
        return _handle_streaming_response(gemini_payload, http_request, request.model)

    return await _handle_non_streaming_response(gemini_payload, http_request, request.model)


def _create_error_response(message: str, status_code: int, error_type: str = "api_error", code: int | None = None) -> Response:
    """Helper to create standardized error responses."""
    return Response(
        content=json.dumps({
            "error": {
                "message": message,
                "type": error_type,
                "code": code or status_code,
            },
        }),
        status_code=status_code,
        media_type="application/json",
    )


def _handle_streaming_response(gemini_payload: dict, http_request: Request, model: str) -> StreamingResponse:
    """Handles the logic for streaming responses."""

    async def openai_stream_generator():
        try:
            response = await send_gemini_request(gemini_payload, http_request.app.state.client_session, is_streaming=True)

            if isinstance(response, StreamingResponse):
                response_id = "chatcmpl-" + str(uuid.uuid4())
                logger.info(f"Starting streaming response: {response_id}")

                async for chunk in response.body_iterator:
                    yield await _process_stream_chunk(chunk, model, response_id)

                # Send the final [DONE] marker
                yield "data: [DONE]\n\n"
                logger.info(f"Completed streaming response: {response_id}")
            else:
                yield _handle_stream_error(response)
        except Exception as e:
            logger.error(f"Streaming error: {e!s}")
            error_data = {"error": {"message": f"Streaming failed: {e!s}", "type": "api_error", "code": 500}}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        openai_stream_generator(),
        media_type="text/event-stream",
    )


async def _process_stream_chunk(chunk: bytes | memoryview | str, model: str, response_id: str) -> str:
    """Process a single chunk of streaming data."""
    if isinstance(chunk, memoryview):
        chunk = chunk.tobytes()
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", "ignore")

    if not chunk.startswith("data: "):
        return ""

    try:
        # Parse the Gemini streaming chunk
        chunk_data = chunk[6:]  # Remove 'data: ' prefix
        gemini_chunk = json.loads(chunk_data)

        # Check if this is an error chunk
        if "error" in gemini_chunk:
            logger.error(f"Error in streaming response: {gemini_chunk['error']}")
            error_data = {
                "error": {
                    "message": gemini_chunk["error"].get("message", "Unknown error"),
                    "type": gemini_chunk["error"].get("type", "api_error"),
                    "code": gemini_chunk["error"].get("code"),
                },
            }
            return f"data: {json.dumps(error_data)}\n\n"

        # Transform to OpenAI format
        openai_chunk = gemini_stream_chunk_to_openai(
            gemini_chunk,
            model,
            response_id,
        )

        # Send as OpenAI streaming format
        await asyncio.sleep(0)
        return f"data: {json.dumps(openai_chunk)}\n\n"

    except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
        logger.warning(f"Failed to parse streaming chunk: {e!s}")
        return ""


def _handle_stream_error(response: Response) -> str:
    """Handle error case where response is not a StreamingResponse."""
    error_msg = "Streaming request failed"
    status_code = 500

    if hasattr(response, "status_code"):
        status_code = response.status_code
        error_msg += f" (status: {status_code})"

    if hasattr(response, "body"):
        try:
            error_body = response.body
            if isinstance(error_body, memoryview):
                error_body = error_body.tobytes()
            if isinstance(error_body, bytes):
                error_body = error_body.decode("utf-8", "ignore")
            error_data = json.loads(error_body)
            if "error" in error_data:
                error_msg = error_data["error"].get("message", error_msg)
        except Exception:
            pass

    logger.error(f"Streaming request failed: {error_msg}")
    error_data = {
        "error": {
            "message": error_msg,
            "type": "invalid_request_error" if status_code == 404 else "api_error",
            "code": status_code,
        },
    }
    return f"data: {json.dumps(error_data)}\n\n"


async def _handle_non_streaming_response(gemini_payload: dict, http_request: Request, model: str) -> Response | dict:
    """Handles the logic for non-streaming responses."""
    try:
        response = await send_gemini_request(gemini_payload, http_request.app.state.client_session, is_streaming=False)

        if isinstance(response, Response) and response.status_code != 200:
            return _process_error_response(response)

        try:
            # Parse Gemini response and transform to OpenAI format
            response_body = response.body
            if isinstance(response_body, memoryview):
                response_body = response_body.tobytes()

            gemini_response = json.loads(response_body)
            openai_response = gemini_response_to_openai(gemini_response, model)

            logger.info(f"Successfully processed non-streaming response for model: {model}")
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse Gemini response: {e!s}")
            return _create_error_response(f"Failed to process response: {e!s}", 500)
        else:
            return openai_response
    except Exception as e:
        logger.error(f"Non-streaming request failed: {e!s}")
        return _create_error_response(f"Request failed: {e!s}", 500)


def _process_error_response(response: Response) -> Response:
    """Extracts error details from a failed Gemini API response."""
    logger.error(f"Gemini API error: status={response.status_code}")
    try:
        error_body = response.body
        if isinstance(error_body, memoryview):
            error_body = error_body.tobytes()
        if isinstance(error_body, bytes):
            error_body = error_body.decode("utf-8", "ignore")

        error_data = json.loads(error_body)
        if "error" in error_data:
            return _create_error_response(
                error_data["error"].get("message", f"API error: {response.status_code}"),
                response.status_code,
                error_data["error"].get("type", "invalid_request_error" if response.status_code == 404 else "api_error"),
                error_data["error"].get("code", response.status_code),
            )
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    return _create_error_response(
        f"API error: {response.status_code}",
        response.status_code,
        "invalid_request_error" if response.status_code == 404 else "api_error",
    )


@router.get("/v1/models")
async def openai_list_models(request: Request):
    """
    OpenAI-compatible models endpoint.
    Returns available models in OpenAI format.
    """

    try:
        logger.info("OpenAI models list requested")

        # Convert our Gemini models to OpenAI format
        openai_models = []
        for model in SUPPORTED_MODELS:
            # Remove "models/" prefix for OpenAI compatibility
            model_id = model["name"].replace("models/", "")
            openai_models.append({
                "id": model_id,
                "object": "model",
                "created": 1677610602,  # Static timestamp
                "owned_by": "google",
                "permission": [
                    {
                        "id": "modelperm-" + model_id.replace("/", "-"),
                        "object": "model_permission",
                        "created": 1677610602,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": False,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False,
                    },
                ],
                "root": model_id,
                "parent": None,
            })
        logger.info(f"Returning {len(openai_models)} models")
    except Exception as e:
        logger.error(f"Failed to list models: {e!s}")
        return _create_error_response(f"Failed to list models: {e!s}", 500)

    return {"object": "list", "data": openai_models}
