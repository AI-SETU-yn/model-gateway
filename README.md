# Model Gateway

Model Gateway exposes the same public inference APIs while routing inference through LiteLLM to an OpenAI-compatible backend such as vLLM.

## Public APIs
- `POST /planner`
- `POST /generate`
- `POST /security/classify`
- `POST /chat/general`

## Provider Architecture
- API contracts remain unchanged.
- Service layer remains unchanged.
- `BaseInferenceProvider` is preserved.
- Active provider wiring now uses `LiteLLMProvider` + `LiteLLMClient`.
- vLLM is accessed through the OpenAI-compatible `/v1/chat/completions` API.

## Configuration
Configuration is environment-driven through `configs/model.yaml` plus `${ENV_NAME:default}` interpolation.

Example:
```yaml
models:
  erp:
    model_name: ${MODEL_GATEWAY_ERP_MODEL_NAME:qwen-erp}
    base_model: ${MODEL_GATEWAY_ERP_BASE_MODEL:Qwen/Qwen2.5-1.5B-Instruct}
    provider:
      type: ${MODEL_GATEWAY_ERP_PROVIDER:vllm}
      api_base: ${MODEL_GATEWAY_ERP_API_BASE:http://127.0.0.1:4000/v1}
      api_key: ${MODEL_GATEWAY_ERP_API_KEY:local}
      deployment_name: ${MODEL_GATEWAY_ERP_DEPLOYMENT:qwen-erp}
```

## Migration Notes
- No changes are required in AI Runtime.
- No endpoint paths changed.
- No request/response schemas changed.
- Planner, generation, security classification, and general chat continue to use the same service layer.
- The former local Transformers execution path has been replaced by a LiteLLM/vLLM-backed provider wiring path.
- Legacy flat provider fields (`base_url`, `api_key`, `api_version`, `organization`, `deployment_name`) are still normalized into the new nested provider config for backward compatibility.

## Backward Compatibility
- Existing API consumers remain compatible.
- Existing service-layer behavior remains compatible.
- Existing response formatting remains compatible.
- Existing token accounting remains compatible because normalized `InferenceResponse.usage` is preserved.
