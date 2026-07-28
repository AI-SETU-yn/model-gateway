import pytest

from app.schemas.inference import GenerateRequest
from app.services.chat_template import ChatTemplateHandler
from app.services.prompt_builder import PromptBuilder


DEFAULT_SYSTEM_PROMPT = 'Use only supplied enterprise data.'


class FakeTokenizer:
    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        self.messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return 'rendered chat prompt'


def test_generate_request_keeps_legacy_prompt_contract() -> None:
    request = GenerateRequest.model_validate({'adapter': 'academic', 'prompt': 'legacy prompt'})

    assert request.adapter == 'academic'
    assert request.prompt == 'legacy prompt'
    assert request.messages is None


def test_generate_request_accepts_future_structured_contract() -> None:
    request = GenerateRequest.model_validate(
        {
            'adapter': 'academic',
            'messages': [{'role': 'user', 'content': 'List academic years'}],
            'toolResult': {'data': [{'academicYear': '2025-2026'}]},
            'responseType': 'structured_summary',
            'generationPolicy': {'useToolResultsOnly': True},
        }
    )

    assert request.prompt is None
    assert request.messages is not None
    assert request.tool_result == {'data': [{'academicYear': '2025-2026'}]}
    assert request.response_type == 'structured_summary'
    assert request.generation_policy is not None


def test_generate_request_requires_prompt_messages_or_tool_result() -> None:
    with pytest.raises(ValueError):
        GenerateRequest.model_validate({'adapter': 'academic'})


def test_prompt_builder_wraps_legacy_prompt_with_system_message() -> None:
    request = GenerateRequest.model_validate({'adapter': 'academic', 'prompt': 'legacy prompt'})
    bundle = PromptBuilder(DEFAULT_SYSTEM_PROMPT).build(request)

    assert bundle.source == 'legacy_prompt'
    assert bundle.messages == [
        {'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT},
        {'role': 'user', 'content': 'legacy prompt'},
    ]


def test_prompt_builder_normalizes_nested_json_tool_result_strings() -> None:
    request = GenerateRequest.model_validate(
        {
            'adapter': 'academic',
            'messages': [{'role': 'user', 'content': 'List academic years'}],
            'toolResult': {'content': [{'type': 'text', 'text': '{"data":[{"academicYear":"2025-2026"}]}'}]},
        }
    )

    bundle = PromptBuilder(DEFAULT_SYSTEM_PROMPT).build(request)

    assert bundle.source == 'structured_messages+tool_result'
    assert bundle.messages[-1]['role'] == 'tool'
    assert '"content"' not in bundle.messages[-1]['content']
    assert '"text"' not in bundle.messages[-1]['content']
    assert '\\"academicYear\\"' not in bundle.messages[-1]['content']
    assert '"academicYear": "2025-2026"' in bundle.messages[-1]['content']


def test_prompt_builder_strips_runtime_transport_wrapper_when_tool_result_contains_data() -> None:
    request = GenerateRequest.model_validate(
        {
            'adapter': 'academic',
            'messages': [{'role': 'user', 'content': 'List academic years'}],
            'toolResult': {
                'tool_name': 'academic.get_all_academic_years_by_branch_id',
                'server': 'vidhya-mcp',
                'status': 'success',
                'success': True,
                'response_type': 'structured',
                'data': {'content': [{'type': 'text', 'text': '{"data":[{"academicYear":"2025-2026"}]}'}]},
                'error': None,
                'tool_execution_latency_ms': 12.3,
            },
        }
    )

    bundle = PromptBuilder(DEFAULT_SYSTEM_PROMPT).build(request)

    assert [message['role'] for message in bundle.messages] == ['system', 'user', 'tool']
    assert '"tool_name"' not in bundle.messages[-1]['content']
    assert '"server"' not in bundle.messages[-1]['content']
    assert '"tool_execution_latency_ms"' not in bundle.messages[-1]['content']
    assert '"academicYear": "2025-2026"' in bundle.messages[-1]['content']


def test_chat_template_handler_uses_tokenizer_template() -> None:
    tokenizer = FakeTokenizer()
    messages = [{'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT}, {'role': 'user', 'content': 'hello'}]

    rendered = ChatTemplateHandler().render(tokenizer, messages, use_chat_template=True)

    assert rendered.text == 'rendered chat prompt'
    assert rendered.used_chat_template is True
    assert tokenizer.messages == messages


def test_chat_template_handler_can_fallback_without_template() -> None:
    messages = [{'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT}, {'role': 'user', 'content': 'hello'}]

    rendered = ChatTemplateHandler().render(object(), messages, use_chat_template=True)

    assert rendered.used_chat_template is False
    assert rendered.text.endswith('ASSISTANT:')
