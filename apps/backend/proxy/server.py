
import os
import json
import logging
import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.requests import Request
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from proxy.translator import (
    convert_anthropic_to_openai_request,
    convert_openai_to_anthropic_response,
    convert_openai_chunk_to_anthropic_events
)

# Configuration
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://routerai.ru/api/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PORT = int(os.environ.get("PROXY_PORT", 8080))
HOST = os.environ.get("PROXY_HOST", "127.0.0.1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def handle_messages(request: Request):
    """
    Handle POST /v1/messages
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    logger.info(f"Received Anthropic request: model={body.get('model')}, stream={body.get('stream')}")

    # Determine API Key
    # Priority: Env Var -> x-api-key header -> authorization header
    api_key = OPENAI_API_KEY
    if not api_key:
        api_key = request.headers.get("x-api-key")
    if not api_key:
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
            
    if not api_key:
        logger.error("No API key found in headers or env")
        return JSONResponse({"error": {"type": "authentication_error", "message": "Missing API Key"}}, status_code=401)

    # Transform request
    openai_payload = convert_anthropic_to_openai_request(body)
    
    # Send to OpenAI provider
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # We use httpx for async requests
    client = httpx.AsyncClient(timeout=120.0)
    
    try:
        if openai_payload.get("stream"):
            return await _handle_streaming_request(client, openai_payload, headers)
        else:
            return await _handle_standard_request(client, openai_payload, headers)
    except Exception as e:
        logger.error(f"Error forwarding request: {e}")
        return JSONResponse({"error": {"type": "server_error", "message": str(e)}}, status_code=500)

async def handle_models(request: Request):
    """
    Handle GET /v1/models (Pass-through)
    """
    api_key = OPENAI_API_KEY or request.headers.get("x-api-key")  # Simple check
    if not api_key:
         return JSONResponse({"data": []}) # Return empty valid if no key, to prevent crash

    url = f"{OPENAI_BASE_URL.rstrip('/')}/models"
    # Fix: remove /chat/completions suffix if present in BASE URL for models call
    # But usually BASE_URL is .../v1. If user set it to .../chat/completions it's wrong.
    # We assume OPENAI_BASE_URL is root of v1. 
    # Current config: https://routerai.ru/api/v1
    
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, headers=headers)
            return JSONResponse(resp.json(), status_code=resp.status_code)
        except Exception as e:
            logger.error(f"Models error: {e}")
            return JSONResponse({"data": []})

async def _handle_standard_request(client, payload, headers):
    url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    response = await client.post(url, json=payload, headers=headers)
    await client.aclose()
    
    if response.status_code != 200:
        logger.error(f"OpenAI Upstream Error: {response.text}")
        return JSONResponse(
            {"error": {"type": "upstream_error", "message": response.text}}, 
            status_code=response.status_code
        )
        
    openai_data = response.json()
    anthropic_data = convert_openai_to_anthropic_response(openai_data)
    return JSONResponse(anthropic_data)

async def _handle_streaming_request(client, payload, headers):
    url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    
    async def stream_generator():
        # State tracker for the translator
        state = {"started": False, "content_started": False}
        
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            if response.status_code != 200:
                # If upstream fails immediately, try to yield an error (though SSE expects specific format)
                # Ideally we should buffer first chunk, but for now log it.
                error_body = await response.aread()
                logger.error(f"OpenAI Stream Error: {error_body.decode()}")
                yield f"event: error\ndata: {json.dumps({'error': {'message': error_body.decode()}})}\n\n"
                return

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        events = convert_openai_chunk_to_anthropic_events(chunk, state)
                        for event in events:
                            yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
                    except json.JSONDecodeError:
                        continue
        
        await client.aclose()

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

async def handle_count_tokens(request: Request):
    """
    Handle POST /v1/messages/count_tokens
    Returns a rough estimate of tokens (char count / 4).
    """
    try:
        body = await request.json()
        messages = body.get("messages", [])
        system = body.get("system", "")
        
        # simple estimation
        text_len = 0
        if isinstance(system, str):
            text_len += len(system)
        
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                text_len += len(content)
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        text_len += len(block.get("text", ""))
        
        # Rough estimate: 4 chars per token
        input_tokens = int(text_len / 4) + len(messages) * 4 # +4 overhead per msg
        
        return JSONResponse({"input_tokens": input_tokens})
    except Exception as e:
        logger.error(f"Count tokens error: {e}")
        return JSONResponse({"input_tokens": 0})

async def handle_event_logging(request: Request):
    """
    Handle POST /api/event_logging/batch
    Swallow telemetry to prevent 404s.
    """
    return JSONResponse({}, status_code=200)

routes = [
    Route("/v1/messages", handle_messages, methods=["POST"]),
    Route("/v1/messages/count_tokens", handle_count_tokens, methods=["POST"]),
    Route("/api/v1/messages", handle_messages, methods=["POST"]),
    Route("/messages", handle_messages, methods=["POST"]),
    Route("/v1/models", handle_models, methods=["GET"]),
    Route("/api/v1/models", handle_models, methods=["GET"]),
    
    # Telemetry stub
    Route("/api/event_logging/batch", handle_event_logging, methods=["POST"]),
]

middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
]

app = Starlette(routes=routes, middleware=middleware)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
