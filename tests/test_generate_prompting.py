import pytest

from app.config.settings import GenerationConfig
from app.schemas.inference import GenerateRequest
from app.services.chat_template import ChatTemplateHandler
from app.services.generation_parameters import GenerationParameters
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


def test_prompt_builder_extracts_business_data_from_legacy_runtime_prompt() -> None:
    legacy_prompt = (
        'You are the Yn AI Setu assistant.\n'
        'Planner intent: academic.academic_year.list\n'
        'Requires tool: True\n'
        'Tool execution result: '
        '{"tool_name":"academic.get_all_academic_years_by_branch_id","server":"vidhya-mcp",'
        '"status":"success","success":true,"response_type":"structured",'
        '"data":{"content":[{"type":"text","text":"{\\"data\\":[{\\"academicYear\\":\\"2025-2026\\"}]}"}]},'
        '"registry_lookup_latency_ms":1.2,"tool_execution_latency_ms":44.5}\n\n'
        'User message:\n'
        'List all academic years'
    )
    request = GenerateRequest.model_validate({'adapter': 'academic', 'prompt': legacy_prompt})

    bundle = PromptBuilder(DEFAULT_SYSTEM_PROMPT).build(request)

    assert bundle.source == 'legacy_prompt+extracted_tool_result'
    assert [message['role'] for message in bundle.messages] == ['system', 'user', 'tool']
    assert bundle.messages[1]['content'] == 'List all academic years'
    assert 'academic.academic_year.list' not in bundle.messages[1]['content']
    assert '"tool_name"' not in bundle.messages[-1]['content']
    assert '"server"' not in bundle.messages[-1]['content']
    assert '"registry_lookup_latency_ms"' not in bundle.messages[-1]['content']
    assert '"academicYear": "2025-2026"' in bundle.messages[-1]['content']


def test_generation_parameters_omit_sampling_kwargs_when_sampling_disabled() -> None:
    tokenizer = type('Tokenizer', (), {'pad_token_id': 0, 'eos_token_id': 1})()
    params = GenerationParameters(
        max_new_tokens=128,
        temperature=0.1,
        top_p=0.9,
        do_sample=False,
        repetition_penalty=1.05,
    )

    kwargs = params.to_model_kwargs(tokenizer)

    assert kwargs['max_new_tokens'] == 128
    assert kwargs['do_sample'] is False
    assert 'temperature' not in kwargs
    assert 'top_p' not in kwargs


def test_generation_parameters_use_planner_specific_token_budget() -> None:
    tokenizer = type('Tokenizer', (), {'pad_token_id': 0, 'eos_token_id': 1})()
    config = GenerationConfig(
        max_new_tokens=128,
        planner_max_new_tokens=96,
        temperature=0.1,
        top_p=0.9,
        do_sample=False,
        repetition_penalty=1.05,
    )

    params = GenerationParameters.from_config(config, planner_mode=True)
    kwargs = params.to_model_kwargs(tokenizer)

    assert kwargs['max_new_tokens'] == 96
    assert kwargs['do_sample'] is False
    assert 'temperature' not in kwargs
    assert 'top_p' not in kwargs


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
