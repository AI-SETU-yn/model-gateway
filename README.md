# Yn AI Setu Model Gateway

Inference-only Model Gateway for YN Setu. This service is independent from AI Runtime and is responsible only for loading the base model, attaching LoRA adapters, and generating model responses.

## Scope

This gateway includes:
- base model loading with Transformers
- LoRA adapter loading with PEFT
- adapter hot reload
- adapter cache
- generation API
- planner JSON generation API
- health and metrics endpoints

This gateway does not include:
- tool execution
- MCP client logic
- memory
- RAG
- vector database access
- business logic
- downstream microservice calls

## Folder Structure

- `app/main.py`: FastAPI bootstrap and lifespan
- `app/api/routers/inference.py`: `/generate` and `/planner`
- `app/api/routers/management.py`: `/health`, `/metrics`, `/adapters`, `/reload-adapter`
- `app/config/settings.py`: environment settings and YAML config loading
- `app/services/adapter_manager.py`: adapter discovery and validation
- `app/services/model_loader.py`: base model loading and adapter cache
- `app/services/inference.py`: text generation service
- `app/services/planner.py`: planner JSON generation service
- `app/services/metrics_service.py`: in-memory metrics collection
- `app/services/dependencies.py`: dependency wiring
- `app/schemas/`: request and response schemas
- `app/exceptions/`: error types and HTTP handlers
- `app/middleware/request_context.py`: request metadata propagation
- `app/utils/logging.py`: structured logging setup
- `app/utils/system.py`: memory usage helpers
- `configs/model.yaml`: model and generation configuration
- `adapters/`: adapter folders such as `academic/`, `hrms/`, `finance/`

## Configuration

Main configuration file: `configs/model.yaml`

Example:

```yaml
base_model: Qwen/Qwen2.5-1.5B-Instruct
default_adapter: academic
adapters_root: adapters
device: auto
dtype: auto
trust_remote_code: true
planner_system_prompt: |
  You are the ERP Planner model for YN Setu.
  Return only valid JSON.
  Extract the user's business intent, tool hint, and parameters.
  Do not include markdown.
generation:
  max_new_tokens: 256
  temperature: 0.1
  top_p: 0.9
  do_sample: false
  repetition_penalty: 1.05
  planner_max_new_tokens: 256
```

Environment overrides are read from `.env` using the `MODEL_GATEWAY_` prefix.

## Adapter Layout

Each adapter directory must contain:
- `adapter_model.safetensors`
- `adapter_config.json`

Example:

```text
adapters/
  academic/
    adapter_model.safetensors
    adapter_config.json
  hrms/
    adapter_model.safetensors
    adapter_config.json
```

## APIs

### POST `/generate`

Request:

```json
{
  "adapter": "academic",
  "prompt": "Show student attendance"
}
```

Response:

```json
{
  "response": "...",
  "adapter": "academic",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "usage": {
    "promptTokens": 12,
    "completionTokens": 35,
    "totalTokens": 47
  },
  "generationTimeMs": 842.17
}
```

### POST `/planner`

Request:

```json
{
  "adapter": "academic",
  "query": "Show my attendance"
}
```

Response:

```json
{
  "intent": "attendance",
  "tool": "academic.getAttendance",
  "parameters": {
    "studentId": "123"
  },
  "rawResponse": "{...}",
  "adapter": "academic",
  "model": "Qwen/Qwen2.5-1.5B-Instruct"
}
```

### POST `/reload-adapter`

Request:

```json
{
  "adapter": "academic"
}
```

Response:

```json
{
  "adapter": "academic",
  "reloaded": true,
  "loadedAdapters": ["academic"]
}
```

### GET `/health`

### GET `/metrics`

### GET `/adapters`

## Startup

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 9000
```

## Docker

```bash
docker build -t yn-setu-model-gateway .
docker run --env-file .env -p 9000:9000 yn-setu-model-gateway
```

## Production Notes

- The base model is loaded only once.
- Adapters are cached and reused.
- Reloading an adapter does not require a server restart.
- Inference is serialized behind a bounded concurrency semaphore for safer GPU execution.
- Structured logs include adapter loads, generation time, token counts, errors, and memory usage.
- The gateway is ready to integrate with AI Runtime through HTTP without sharing runtime internals.
