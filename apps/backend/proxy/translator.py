
import json
import logging
import time

logger = logging.getLogger(__name__)

def convert_anthropic_to_openai_request(anthropic_payload: dict, model_override: str = None) -> dict:
    """
    Convert an Anthropic /v1/messages request to an OpenAI /v1/chat/completions request.
    
    Args:
        anthropic_payload: The JSON body of the Anthropic request.
        model_override: Optional model ID to force (e.g., if passed via env var).
    
    Returns:
        A dict representing the OpenAI request body.
    """
    messages = []
    
    # 1. Handle System Prompt
    system_prompt = anthropic_payload.get("system")
    if system_prompt:
        # Anthropic 'system' can be string or list of dicts (uncommon but possible in some SDK versions)
        # We'll handle the standard string case or list of text blocks
        content = ""
        if isinstance(system_prompt, str):
            content = system_prompt
        elif isinstance(system_prompt, list):
            # Extract text from blocks
            content = "".join([block.get("text", "") for block in system_prompt if block.get("type") == "text"])
        
        if content:
            messages.append({"role": "system", "content": content})
            
    # 2. Handle Messages
    for msg in anthropic_payload.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        
        # Anthropic content can be string or list of blocks
        openai_content = ""
        if isinstance(content, str):
            openai_content = content
        elif isinstance(content, list):
            # Concatenate text blocks. Image blocks not yet supported for basic proxy.
            parts = []
            for block in content:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                # TODO: Handle image blocks if RouterAI supports them in OpenAI format
            openai_content = "".join(parts)
            
        messages.append({"role": role, "content": openai_content})

    openai_payload = {
        "model": model_override or anthropic_payload.get("model"),
        "messages": messages,
        "stream": anthropic_payload.get("stream", False),
    }

    # 3. Map Optional Parameters
    if "max_tokens" in anthropic_payload:
        openai_payload["max_tokens"] = anthropic_payload["max_tokens"]
        
    if "temperature" in anthropic_payload:
        openai_payload["temperature"] = anthropic_payload["temperature"]
        
    if "top_p" in anthropic_payload:
        openai_payload["top_p"] = anthropic_payload["top_p"]

    if "stop_sequences" in anthropic_payload:
        openai_payload["stop"] = anthropic_payload["stop_sequences"]

    return openai_payload


def convert_openai_to_anthropic_response(openai_response: dict) -> dict:
    """
    Convert a non-streaming OpenAI response to an Anthropic response.
    """
    choice = openai_response.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")
    
    anthropic_response = {
        "id": openai_response.get("id"),
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": content
            }
        ],
        "model": openai_response.get("model"),
        "stop_reason": _map_stop_reason(choice.get("finish_reason")),
        "stop_sequence": None, # OpenAI doesn't always provide the exact stop sequence
        "usage": {
            "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_response.get("usage", {}).get("completion_tokens", 0)
        }
    }
    return anthropic_response

def convert_openai_chunk_to_anthropic_events(openai_chunk: dict, previous_state: dict = None) -> list[dict]:
    """
    Convert an OpenAI streaming chunk into one or more Anthropic SSE events.
    Anthropic streaming is more verbose (message_start, content_block_start, delta, content_block_stop, message_stop).
    
    Returns a list of event dicts to be sent as SEE events.
    """
    events = []
    choice = openai_chunk.get("choices", [{}])[0]
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")
    
    # Check if this is the start of the stream (we might need a state tracker if we were doing this perfectly, 
    # but often we can just infer or send message_start first if we haven't seen it)
    # For simplicity, we assume the caller handles the very first 'message_start' event logic or we handle it lazily.
    # ACTUALLY: Anthropic expects a strict sequence.
    # 1. message_start
    # 2. content_block_start
    # ... deltas ...
    # 3. content_block_stop
    # 4. message_delta (stop_reason)
    # 5. message_stop
    
    # We will need some state passed in `previous_state` dict to know if we've sent start events.
    if previous_state is None:
        previous_state = {"started": False, "content_started": False}

    if not previous_state.get("started"):
        # Emit message_start
        events.append({
            "event": "message_start",
            "data": {
                "type": "message_start",
                "message": {
                    "id": openai_chunk.get("id", f"msg_{int(time.time())}"),
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": openai_chunk.get("model"),
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0} # Placeholder
                }
            }
        })
        previous_state["started"] = True

    content_text = delta.get("content", "")
    
    if content_text:
        if not previous_state.get("content_started"):
             events.append({
                "event": "content_block_start",
                "data": {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""}
                }
             })
             previous_state["content_started"] = True
        
        events.append({
            "event": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": content_text}
            }
        })

    if finish_reason:
        if previous_state.get("content_started"):
            events.append({
                "event": "content_block_stop",
                "data": {"type": "content_block_stop", "index": 0}
            })
            
        events.append({
            "event": "message_delta",
            "data": {
                "type": "message_delta",
                "delta": {
                    "stop_reason": _map_stop_reason(finish_reason),
                    "stop_sequence": None 
                },
                "usage": {"output_tokens": 0} # We don't have exact count here easily without counting
            }
        })
        events.append({
            "event": "message_stop",
            "data": {"type": "message_stop"}
        })

    return events

def _map_stop_reason(openai_reason: str) -> str:
    if openai_reason == "stop":
        return "end_turn"
    elif openai_reason == "length":
        return "max_tokens"
    elif openai_reason == "tool_calls":
        return "tool_use" # Not fully supported in this simple proxy yet
    return "end_turn" # Default
