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


def test_generate_request_accepts_structured_response_contract() -> None:
    request = GenerateRequest.model_validate(
        {
            'adapter': 'academic',
            'messages': [{'role': 'user', 'content': 'List academic years'}],
            'toolResult': {'data': [{'academicYear': '2025-2026'}]},
            'responseType': 'enterprise',
            'generationPolicy': {
                'grounded': True,
                'hallucination': 'forbid',
                'format': 'markdown',
                'useToolResultsOnly': True,
            },
            'conversation': {
                'userQuestion': 'List academic years',
                'plannerIntent': 'academic.academic_year.list',
                'executionPlan': [{'step_id': 'step_1', 'intent': 'academic.academic_year.list'}],
            },
            'metadata': {'runtime': 'ai-runtime-v2'},
            'missingParameters': ['academic_year_id'],
        }
    )

    assert request.messages is not None
    assert request.tool_result == {'data': [{'academicYear': '2025-2026'}]}
    assert request.response_type == 'enterprise'
    assert request.generation_policy is not None
    assert request.generation_policy.output_format == 'markdown'
    assert request.conversation is not None
    assert request.conversation.planner_intent == 'academic.academic_year.list'
    assert request.metadata == {'runtime': 'ai-runtime-v2'}
    assert request.missing_parameters == ['academic_year_id']


def test_generate_request_requires_messages_or_tool_result() -> None:
    with pytest.raises(ValueError, match='messages or toolResult'):
        GenerateRequest.model_validate({'adapter': 'academic'})


def test_prompt_builder_wraps_structured_messages_with_system_message() -> None:
    request = GenerateRequest.model_validate({
        'adapter': 'academic',
        'messages': [{'role': 'user', 'content': 'structured message'}],
    })
    bundle = PromptBuilder(DEFAULT_SYSTEM_PROMPT).build(request)

    assert bundle.source == 'structured_messages'
    assert bundle.messages == [
        {'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT},
        {'role': 'user', 'content': 'structured message'},
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


def test_prompt_builder_adds_structured_response_context() -> None:
    request = GenerateRequest.model_validate(
        {
            'adapter': 'academic',
            'messages': [{'role': 'user', 'content': 'Which value is missing?'}],
            'responseType': 'clarification',
            'toolResult': {'status': 'requires_input'},
            'conversation': {'userQuestion': 'Which value is missing?', 'plannerIntent': 'academic.item.list'},
            'metadata': {'response_type': 'clarification'},
            'missingParameters': ['item_id'],
        }
    )

    bundle = PromptBuilder(DEFAULT_SYSTEM_PROMPT).build(request)

    assert [message['role'] for message in bundle.messages] == ['system', 'user', 'tool', 'tool']
    assert 'Target response type: clarification.' in bundle.messages[0]['content']
    assert '"missingParameters": [' in bundle.messages[2]['content']
    assert '"item_id"' in bundle.messages[2]['content']


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


def test_prompt_builder_strips_transport_metadata_from_structured_runtime_result() -> None:
    request = GenerateRequest.model_validate(
        {
            'adapter': 'academic',
            'messages': [{'role': 'user', 'content': 'List all academic years'}],
            'toolResult': {
                'tool_name': 'academic.get_all_academic_years_by_branch_id',
                'server': 'vidhya-mcp',
                'status': 'success',
                'success': True,
                'response_type': 'structured',
                'data': {'content': [{'type': 'text', 'text': '{"data":[{"academicYear":"2025-2026"}]}'}]},
                'registry_lookup_latency_ms': 1.2,
                'tool_execution_latency_ms': 44.5,
            },
        }
    )

    bundle = PromptBuilder(DEFAULT_SYSTEM_PROMPT).build(request)

    assert bundle.source == 'structured_messages+tool_result'
    assert [message['role'] for message in bundle.messages] == ['system', 'user', 'tool']
    assert bundle.messages[1]['content'] == 'List all academic years'
    assert '"tool_name"' not in bundle.messages[-1]['content']
    assert '"server"' not in bundle.messages[-1]['content']
    assert '"registry_lookup_latency_ms"' not in bundle.messages[-1]['content']
    assert '"academicYear": "2025-2026"' in bundle.messages[-1]['content']


def test_prompt_builder_builds_multi_tool_context_from_visible_steps() -> None:
    request = GenerateRequest.model_validate(
        {
            'adapter': 'academic',
            'messages': [{'role': 'user', 'content': 'List two datasets'}],
            'responseType': 'multi_tool',
            'conversation': {
                'userQuestion': 'List two datasets',
                'executionPlan': [
                    {'step_id': 'step_1', 'visible_in_response': True},
                    {'step_id': 'step_2', 'visible_in_response': True},
                ],
            },
            'toolResult': {
                'success': True,
                'steps': [
                    {
                        'step_id': 'step_1',
                        'result': {
                            'tool_name': 'generic.first',
                            'server': 'mcp',
                            'success': True,
                            'data': {'items': [{'name': 'Alpha'}]},
                        },
                    },
                    {
                        'step_id': 'step_2',
                        'result': {
                            'tool_name': 'generic.second',
                            'server': 'mcp',
                            'success': True,
                            'data': {'items': [{'name': 'Beta'}]},
                        },
                    },
                ],
            },
        }
    )

    bundle = PromptBuilder(DEFAULT_SYSTEM_PROMPT).build(request)

    assert '"results": [' in bundle.messages[-1]['content']
    assert '"name": "Alpha"' in bundle.messages[-1]['content']
    assert '"name": "Beta"' in bundle.messages[-1]['content']
    assert '"tool_name"' not in bundle.messages[-1]['content']
    assert '"server"' not in bundle.messages[-1]['content']


def test_prompt_builder_hides_invisible_multi_tool_helper_steps() -> None:
    request = GenerateRequest.model_validate(
        {
            'adapter': 'academic',
            'messages': [{'role': 'user', 'content': 'List dependent result'}],
            'responseType': 'multi_tool',
            'conversation': {
                'userQuestion': 'List dependent result',
                'executionPlan': [
                    {'step_id': 'step_1', 'visible_in_response': False},
                    {'step_id': 'step_2', 'visible_in_response': True},
                ],
            },
            'toolResult': {
                'success': True,
                'steps': [
                    {'step_id': 'step_1', 'result': {'success': True, 'data': {'items': [{'name': 'Alpha'}]}}},
                    {'step_id': 'step_2', 'result': {'success': True, 'data': {'items': [{'name': 'Beta'}]}}},
                ],
                'data': {'items': [{'name': 'Beta'}]},
            },
        }
    )

    bundle = PromptBuilder(DEFAULT_SYSTEM_PROMPT).build(request)

    assert '"name": "Beta"' in bundle.messages[-1]['content']
    assert '"name": "Alpha"' not in bundle.messages[-1]['content']


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
