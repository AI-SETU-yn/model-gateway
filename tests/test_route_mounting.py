"""Route mounting verification (guardrail audit finding #6).

`app/main.py` only mounts `app/api/chat.py` (`/chat`, `/v1/models`) and
`app/api/health.py` (`/health*`) -- the LiteLLM-backed path. It never mounts
`app/api/routers/inference.py`, which is where `/generate`, `/planner`, and
`/security/classify` are defined (exactly the paths AI Runtime's
`ModelGatewayClient` and `SecurityClassifierClient` call). These tests
confirm that state directly against the assembled route table, without
sending real HTTP traffic or requiring a live LiteLLM backend.

Note: `app/api/routers/inference.py` cannot simply be mounted to close this
gap -- its dependency providers (`get_generation_service`,
`get_planner_service`, `get_security_classifier_service`, ...) do not exist
in `app/services/dependencies.py`, and the service modules it wires import
`ModelProfileConfig` / `ModelGatewayConfig` / `GenerationConfig` from
`app.config.settings`, which no longer defines them (`app/config/settings.py`
is now `from app.core.config import *`). Importing that router raises
`ImportError` today; see the final report for details.

Required settings are exported *before* `app.main` is imported: `app/main.py`
builds its FastAPI app at module import time (`app = create_app()`), so a
per-test `monkeypatch.setenv` fixture would run too late to affect it.
"""

from __future__ import annotations

import os

os.environ.setdefault('LITELLM_BASE_URL', 'http://localhost:4000')
os.environ.setdefault('DEFAULT_MODEL', 'qwen-erp')
os.environ.setdefault('REQUEST_TIMEOUT', '30')
os.environ.setdefault('MAX_RETRIES', '2')
os.environ.setdefault('MODEL_REGISTRY_PATH', 'configs/models.yaml')

import pytest

from app.main import app as gateway_app  # noqa: E402 - env vars must be set first


def _routed_paths(app) -> set[str]:
    return {route.path for route in app.routes if hasattr(route, 'path')}


def test_live_litellm_routes_are_mounted():
    paths = _routed_paths(gateway_app)
    assert '/chat' in paths
    assert '/v1/models' in paths
    assert '/health/live' in paths
    assert '/health/ready' in paths
    assert '/health' in paths


def test_legacy_planner_generate_security_routes_are_not_mounted():
    """Documents the confirmed audit finding: these paths are unreachable today."""

    paths = _routed_paths(gateway_app)
    assert '/planner' not in paths
    assert '/generate' not in paths
    assert '/security/classify' not in paths


def test_inference_router_is_not_importable_with_current_dependency_wiring():
    """Explains *why* the router isn't mounted: it can't be, safely, as-is."""

    with pytest.raises(ImportError):
        import app.api.routers.inference  # noqa: F401
