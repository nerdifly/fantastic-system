# main.py
import json
import time
import httpx # Use httpx for async requests
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, Response # Import Response for OPTIONS
from fastapi.middleware.cors import CORSMiddleware # Import CORS Middleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any

# --- Configuration ---
AUTH_TOKEN_FILE = 'auth_token.json'
PUTER_SIGNUP_URL = "https://puter.com/signup"
PUTER_API_BASE_URL = "https://api.puter.com"
DEBUG = True # Set to False for production
MAX_RETRIES_ON_USAGE_LIMIT = 1 # Number of times to retry after hitting usage limit

# Define the list of supported models for the /v1/models endpoint
SUPPORTED_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "o3",
    "o3-mini",
    "o4-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4.5-preview"
]

# --- Pydantic Models ---

# Models for /v1/chat/completions Request
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

# Models for /v1/chat/completions Non-Streaming Response
class ResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ResponseChoice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None

# Models for /v1/chat/completions Streaming Response
class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None

class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]
    system_fingerprint: Optional[str] = None
    usage: Optional[Usage] = None

# --- Models for /v1/models Endpoint ---
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time())) # Or use a fixed timestamp
    owned_by: str = "system" # Or "openai", "puter-proxy"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

# --- Authentication and Helper Functions (adapted for async - unchanged) ---
# ... (keep all functions: read_auth_token, set_auth_token_working, signup_user, get_user_app_token, create_puter_request_body) ...
async def read_auth_token():
    try:
        if not os.path.exists(AUTH_TOKEN_FILE):
            return None
        with open(AUTH_TOKEN_FILE, 'r') as token_file:
            try:
                data = json.load(token_file)
            except json.JSONDecodeError:
                return None

            if not data or "auth" not in data or "tokens" not in data["auth"]:
                return None

            for token_entry in data["auth"]["tokens"]:
                if token_entry.get("working"):
                    return token_entry.get("token")
            return None
    except FileNotFoundError:
        return None

async def set_auth_token_working(token: str, working: bool):
    data = {"auth": {"tokens": []}}
    try:
        if os.path.exists(AUTH_TOKEN_FILE):
            with open(AUTH_TOKEN_FILE, 'r') as token_file:
                try:
                    data = json.load(token_file)
                    if "auth" not in data: data["auth"] = {"tokens": []}
                    if "tokens" not in data["auth"]: data["auth"]["tokens"] = []
                except json.JSONDecodeError:
                    data = {"auth": {"tokens": []}}
    except FileNotFoundError:
        pass

    token_found = False
    for token_entry in data["auth"]["tokens"]:
        if token_entry.get("token") == token:
            token_entry["working"] = working
            token_found = True
            break

    if not token_found and working: # Only add new tokens if they are marked as working
        data["auth"]["tokens"].append({"token": token, "working": working})

    with open(AUTH_TOKEN_FILE, 'w') as token_file:
        json.dump(data, token_file, indent=2)

async def signup_user(client: httpx.AsyncClient):
    print("Attempting to sign up a new temporary user...")
    headers_signup = {
        "Content-Type": "application/json",
        "accept": "*/*",
        "origin": "https://puter.com",
        "referer": "https://puter.com/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    }
    body_signup = {"is_temp": True}
    try:
        response_signup = await client.post(PUTER_SIGNUP_URL, headers=headers_signup, json=body_signup, timeout=30.0)
        response_signup.raise_for_status()

        response_data = response_signup.json()
        auth_token = response_data.get("token")
        if not auth_token:
            print("Signup response did not contain a token.")
            return None

        print(f"Signup successful. New auth token received (first 5 chars): {auth_token[:5]}...")
        # Add the new token and mark it as working
        await set_auth_token_working(auth_token, True)
        return auth_token

    except httpx.RequestError as e:
        print(f"Error during signup request: {e}")
        return None
    except httpx.HTTPStatusError as e:
        print(f"Signup failed with status {e.response.status_code}: {e.response.text}")
        return None
    except json.JSONDecodeError:
        print(f"Failed to decode JSON response from signup.")
        return None

async def get_user_app_token(client: httpx.AsyncClient, auth_token: str):
    print(f"Getting user app token for auth token (first 5): {auth_token[:5]}...")
    url = f"{PUTER_API_BASE_URL}/auth/get-user-app-token"
    headers = {
        "authorization": f"Bearer {auth_token}",
        "accept": "*/*",
        "content-type": "application/json",
        "origin": "http://localhost:5500",
        "referer": "http://localhost:5500/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    }
    body_auth = {"origin": "http://localhost:5500"} # Adjust if needed
    try:
        response = await client.post(url, headers=headers, json=body_auth, timeout=15.0)
        response.raise_for_status()

        response_data = response.json()
        new_token = response_data.get("token")
        if not new_token:
            print("Get user app token response did not contain a token.")
            return None

        print(f"User app token received (first 5 chars): {new_token[:5]}...")
        return new_token

    except httpx.RequestError as e:
        print(f"Error during get_user_app_token request: {e}")
        return None
    except httpx.HTTPStatusError as e:
        print(f"Get user app token failed with status {e.response.status_code}: {e.response.text}")
        if e.response.status_code in [401, 403]:
            print("Marking current auth token as not working due to error.")
            await set_auth_token_working(auth_token, False)
        return None
    except json.JSONDecodeError:
        print(f"Failed to decode JSON response from get_user_app_token.")
        return None

def create_puter_request_body(api_input: ChatCompletionRequest):
    messages_dict = [msg.model_dump() for msg in api_input.messages] # Use model_dump() for Pydantic v2+
    return {
        "interface": "puter-chat-completion",
        "driver": "openai-completion",
        "test_mode": False,
        "method": "complete",
        "args": {
            "messages": messages_dict,
            "model": api_input.model,
            "stream": api_input.stream,
            "temperature": api_input.temperature,
            "top_p": api_input.top_p,
            "max_tokens": api_input.max_tokens,
            # Add others if needed
        }
    }
# --- FastAPI App Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(timeout=None) # Use context default timeouts or set specific ones per request
    print("HTTPX client created.")
    yield
    await app.state.http_client.aclose()
    print("HTTPX client closed.")

app = FastAPI(lifespan=lifespan)

# --- CORS Middleware Configuration ---
origins = ["*"] # Allows all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"], # Allow GET, POST, and OPTIONS
    allow_headers=["*"], # Allow all headers, including Content-Type, Authorization etc.
)

# --- Helper to get valid tokens, including signup ---
async def get_valid_tokens(client: httpx.AsyncClient, attempt_signup=True):
    """Attempts to get a valid auth_token and user_app_token.
       If attempt_signup is True, it will try signing up if no working token is found.
    """
    auth_token = await read_auth_token()
    user_app_token = None

    if auth_token:
        user_app_token = await get_user_app_token(client, auth_token)
        if user_app_token:
            return auth_token, user_app_token # Found working pair

    # If we are here, either no auth_token, or get_user_app_token failed (and marked it bad)
    if attempt_signup:
        print("No valid working token pair found. Attempting signup...")
        auth_token = await signup_user(client)
        if auth_token:
            user_app_token = await get_user_app_token(client, auth_token)
            if user_app_token:
                return auth_token, user_app_token # Success after signup
        # If signup or subsequent get_user_app_token failed
        print("Failed to obtain valid tokens even after signup attempt.")
        return None, None
    else:
        # We didn't attempt signup, and the initial check failed
        print("No valid working token pair found, and signup was not attempted.")
        return None, None


# --- Streaming Response Generator ---
async def stream_puter_response(client: httpx.AsyncClient, stream_response: httpx.Response, original_model: str):
    """Generator function that yields formatted SSE chunks from an active stream response."""
    # NO finally block here to explicitly close the stream - let the context manager in the caller handle it.
    try:
        async for line in stream_response.aiter_lines():
            if line.strip():
                try:
                    data = json.loads(line)
                    if data.get("type") == "text":
                        content = data.get("text", "")
                        if content:
                            chunk = ChatCompletionChunk(
                                model=original_model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0,
                                        delta=ChatCompletionChunkDelta(role="assistant", content=content),
                                        finish_reason=None
                                    )
                                ]
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n" # Use model_dump_json()
                    # Handle other potential message types from Puter if needed
                except json.JSONDecodeError:
                    print(f"Warning: Received non-JSON line from stream: {line}")
                    continue
                except Exception as e:
                    print(f"Error processing stream line: {e}") # Log error during processing
                    # Optionally yield an error chunk here if needed, but avoid breaking the loop unless necessary
                    # yield f"data: {json.dumps({'error': {'message': f'Error processing stream line: {e}', 'code': 500}})}\n\n"
                    continue # Continue processing next line if possible

        # Send the final [DONE] message after successfully iterating through all lines
        yield "data: [DONE]\n\n"

    except httpx.StreamError as e: # More specific error handling for stream issues
        print(f"StreamError during iteration: {e}")
        yield f"data: {json.dumps({'error': {'message': f'Stream error during processing: {e}', 'code': 500}})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        # This catches errors during the initial setup of the iteration or other unexpected issues
        print(f"An unexpected error occurred during stream iteration setup or processing: {e}")
        yield f"data: {json.dumps({'error': {'message': f'Internal server error during stream processing: {e}', 'code': 500}})}\n\n"
        yield "data: [DONE]\n\n"
    # Removed the finally block that called await stream_response.aclose()


# --- API Endpoints ---

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    Returns a list of the models potentially available through this proxy.
    Note: Actual availability depends on the backend Puter service configuration.
    """
    model_cards = [
        ModelCard(id=model_id, created=1677610000) # Using a fixed timestamp for simplicity
        for model_id in SUPPORTED_MODELS
    ]
    return ModelList(data=model_cards)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, fastapi_request: Request):
    """
    Mimics the OpenAI Chat Completions endpoint with retry on usage limit.
    """
    client: httpx.AsyncClient = fastapi_request.app.state.http_client

    # Check if requested model is in our list (optional, but good practice)
    # if request.model not in SUPPORTED_MODELS:
    #     raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found or not supported by this proxy.")

    puter_request_body = create_puter_request_body(request)
    current_auth_token = None
    current_user_app_token = None

    for attempt in range(MAX_RETRIES_ON_USAGE_LIMIT + 1):
        # --- Get Tokens ---
        # On retry (attempt > 0), force signup attempt if needed.
        should_signup = (attempt > 0)
        print(f"\nAttempt {attempt + 1} to get tokens (signup allowed: {should_signup})...")
        current_auth_token, current_user_app_token = await get_valid_tokens(client, attempt_signup=True) # Always allow signup attempt here

        if not current_auth_token or not current_user_app_token:
            if attempt < MAX_RETRIES_ON_USAGE_LIMIT:
                 print("Failed to get tokens, will retry if possible.")
                 await asyncio.sleep(1) # Optional small delay before retry
                 continue # Go to next attempt loop iteration
            else:
                 print("Failed to obtain necessary authentication tokens even after retries.")
                 raise HTTPException(status_code=503, detail="Failed to obtain necessary authentication tokens from Puter service after retries.")

        print(f"Attempt {attempt + 1}: Using auth token {current_auth_token[:5]}... and user app token {current_user_app_token[:5]}...")

        # --- Prepare Call ---
        url_call = f"{PUTER_API_BASE_URL}/drivers/call"
        headers_call = {
            "authorization": f"Bearer {current_user_app_token}",
            "accept": "text/event-stream" if request.stream else "*/*",
            "content-type": "application/json;charset=UTF-8",
            "origin": "http://localhost:5500", # Adjust if necessary
            "referer": "http://localhost:5500/", # Adjust if necessary
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }

        # --- Execute Call ---
        try:
            if request.stream:
                # Open the stream context manager
                async with client.stream("POST", url_call, headers=headers_call, json=puter_request_body, timeout=300.0) as response:
                    # Check initial response status IMMEDIATELY after opening stream
                    if response.status_code == 400:
                        # Read the body to check for specific error *without* consuming stream for iteration
                        body_bytes = await response.aread()
                        try:
                            response_json = json.loads(body_bytes.decode())
                            if response_json.get("success") is False and response_json.get("error", {}).get("delegate") == "usage-limited-chat":
                                print(f"Attempt {attempt + 1}: Puter usage limit reached for auth token {current_auth_token[:5]}...")
                                await set_auth_token_working(current_auth_token, False)
                                # Check if more retries are allowed
                                if attempt < MAX_RETRIES_ON_USAGE_LIMIT:
                                     print("Attempting retry with new token...")
                                     await response.aclose() # Close the failed stream response
                                     continue # Go to the next iteration of the for loop to retry
                                else:
                                     print("Max retries reached after usage limit error.")
                                     raise HTTPException(status_code=429, detail="Puter usage limit reached, and max retries exceeded.")
                        except json.JSONDecodeError:
                            pass # Ignore if not JSON, treat as generic 400 below

                    # If status is not 200 or the specific 400 handled above, raise error
                    response.raise_for_status() # Raises for non-2xx/3xx status codes

                    # If successful (status 200 OK), return the StreamingResponse
                    print(f"Attempt {attempt + 1}: Stream connection successful (Status: {response.status_code}). Starting stream...")
                    # IMPORTANT: Pass the already opened `response` object to the generator
                    return StreamingResponse(
                        stream_puter_response(client, response, request.model),
                        media_type="text/event-stream"
                    )

            else: # Non-streaming request
                response_call = await client.post(url_call, headers=headers_call, json=puter_request_body, timeout=120.0)

                if DEBUG:
                    print(f"\nPuter Non-Stream Response Status (Attempt {attempt + 1}): {response_call.status_code}")

                # Check for usage limit error specifically for non-streaming
                if response_call.status_code == 400:
                    try:
                        response_json = response_call.json()
                        if response_json.get("success") is False and response_json.get("error", {}).get("delegate") == "usage-limited-chat":
                            print(f"Attempt {attempt + 1}: Puter usage limit reached for auth token {current_auth_token[:5]}...")
                            await set_auth_token_working(current_auth_token, False)
                            if attempt < MAX_RETRIES_ON_USAGE_LIMIT:
                                print("Attempting retry with new token...")
                                continue # Go to the next iteration of the for loop
                            else:
                                print("Max retries reached after usage limit error.")
                                raise HTTPException(status_code=429, detail="Puter usage limit reached, and max retries exceeded.")
                    except json.JSONDecodeError:
                        pass # Treat as generic 400 below

                # Raise for other errors or if the usage limit wasn't the cause of 400
                response_call.raise_for_status()

                response_json = response_call.json()
                if DEBUG:
                    print(f"Puter Non-Stream Response Body (Attempt {attempt + 1}):")
                    print(json.dumps(response_json, indent=2))

                # --- Format successful Non-Streaming Response ---
                puter_result = response_json.get("result", {})
                puter_message = puter_result.get("message", {})
                puter_content = puter_message.get("content", "")
                puter_finish_reason = puter_result.get("finish_reason")
                prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
                puter_usage = puter_result.get("usage", [])
                if isinstance(puter_usage, list):
                    for item in puter_usage:
                        if item.get('type') == 'prompt': prompt_tokens += item.get('amount', 0)
                        elif item.get('type') == 'completion': completion_tokens += item.get('amount', 0)
                    total_tokens = prompt_tokens + completion_tokens

                output = ChatCompletionResponse(
                    model=request.model,
                    choices=[
                        ResponseChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=puter_content),
                            finish_reason=puter_finish_reason
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens
                    ) if total_tokens > 0 else None
                )
                print(f"Attempt {attempt + 1}: Non-stream request successful.")
                return output # Return the successful response

        # --- Handle Errors during the Call ---
        except httpx.RequestError as e:
             print(f"Attempt {attempt + 1}: HTTP request to Puter failed: {e}")
             # Don't retry network errors immediately, raise final error
             raise HTTPException(status_code=502, detail=f"Failed to communicate with the backend Puter service: {e}")
        except httpx.HTTPStatusError as e:
             print(f"Attempt {attempt + 1}: Puter API returned error status {e.response.status_code}: {e.response.text}")
             # Check if auth token might be invalid (e.g., 401/403) even if not usage limit
             if e.response.status_code in [401, 403] and current_auth_token:
                  print("Marking potentially invalid auth token as non-working.")
                  await set_auth_token_working(current_auth_token, False)
                  # Decide if retry makes sense here. For now, let's just raise.
                  # If you wanted to retry 401/403, you could `continue` here if attempt < MAX_RETRIES...

             # If it's an error we didn't specifically handle for retry, and we've used all retries
             if attempt >= MAX_RETRIES_ON_USAGE_LIMIT:
                  raise HTTPException(status_code=502, detail=f"Backend Puter service returned error after retries: {e.response.status_code} - {e.response.text}")
             else:
                  # If it's an unexpected error but we still have retries, log and continue
                  # This might be too aggressive - depends if you expect retries to fix these
                  print(f"Unexpected HTTP status {e.response.status_code}, but retrying...")
                  await asyncio.sleep(1) # Optional delay
                  continue

        except Exception as e:
             print(f"Attempt {attempt + 1}: An unexpected error occurred: {e}")
             # Don't retry unknown errors, raise final error
             raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

    # If the loop finishes without returning/raising (shouldn't happen with the logic above)
    raise HTTPException(status_code=500, detail="Request processing failed after all attempts.")


# --- Run the server ---
if __name__ == "__main__":
    import uvicorn
    if not os.path.exists(AUTH_TOKEN_FILE):
        print(f"Creating empty auth token file: {AUTH_TOKEN_FILE}")
        with open(AUTH_TOKEN_FILE, 'w') as f:
            json.dump({"auth": {"tokens": []}}, f)

    print("Starting FastAPI server...")
    print("Chat completions endpoint available at: http://127.0.0.1:8000/v1/chat/completions")
    print("Models endpoint available at: http://127.0.0.1:8000/v1/models")
    print("CORS enabled for origins:", origins)
    print("OpenAPI Spec available at: http://127.0.0.1:8000/docs")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)