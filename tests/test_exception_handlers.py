import asyncio

from starlette.requests import Request

from app.exceptions.errors import PlannerResponseError
from app.exceptions.handlers import app_exception_handler


def test_app_exception_handler_does_not_use_reserved_logging_message_key() -> None:
    scope = {
        'type': 'http',
        'method': 'POST',
        'path': '/planner',
        'headers': [],
        'server': ('testserver', 80),
        'scheme': 'http',
        'client': ('testclient', 123),
    }
    request = Request(scope)
    request.state.request_id = 'req-1'
    request.state.correlation_id = 'corr-1'

    response = asyncio.run(app_exception_handler(request, PlannerResponseError('Planner model did not return valid JSON.')))

    assert response.status_code == 502
