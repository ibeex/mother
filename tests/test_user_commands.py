"""Tests for user command parsing and bash execution records."""

from datetime import datetime

from mother.bash_execution import BashExecution, format_for_context, format_for_display
from mother.user_commands import (
    AgentModeCommand,
    CouncilCommand,
    ModelsCommand,
    NormalPrompt,
    QuitAppCommand,
    ReasoningCommand,
    SaveSessionCommand,
    ShellCommand,
    current_model_query,
    current_reasoning_query,
    is_council_multiline_input,
    parse_user_input,
    should_expand_models_query,
    should_expand_reasoning_query,
    should_submit_on_enter,
)


def test_detect_bang_command():
    result = parse_user_input("!ls -la")
    assert isinstance(result, ShellCommand)
    assert result.command == "ls -la"
    assert result.include_in_context is True


def test_detect_double_bang():
    result = parse_user_input("!!ps aux")
    assert isinstance(result, ShellCommand)
    assert result.command == "ps aux"
    assert result.include_in_context is False


def test_detect_normal_prompt():
    result = parse_user_input("hello")
    assert isinstance(result, NormalPrompt)
    assert result.text == "hello"


def test_detect_save_command():
    result = parse_user_input("/save")
    assert isinstance(result, SaveSessionCommand)
    assert result.command == "/save"


def test_detect_export_command():
    result = parse_user_input(" /export ")
    assert isinstance(result, SaveSessionCommand)
    assert result.command == "/export"


def test_detect_quit_command():
    result = parse_user_input("/quit")
    assert isinstance(result, QuitAppCommand)
    assert result.command == "/quit"


def test_detect_exit_command():
    result = parse_user_input(" /exit ")
    assert isinstance(result, QuitAppCommand)
    assert result.command == "/exit"


def test_detect_agent_mode_command():
    result = parse_user_input("/agent")
    assert isinstance(result, AgentModeCommand)
    assert result.command == "/agent"
    assert result.mode is None


def test_detect_standard_agent_mode_command():
    result = parse_user_input(" /agent standard ")
    assert isinstance(result, AgentModeCommand)
    assert result.command == "/agent"
    assert result.mode == "standard"


def test_detect_conversational_agent_mode_command():
    result = parse_user_input(" /agent conversational ")
    assert isinstance(result, AgentModeCommand)
    assert result.command == "/agent"
    assert result.mode == "conversational"


def test_detect_deep_research_agent_mode_command():
    result = parse_user_input(" /agent deep research ")
    assert isinstance(result, AgentModeCommand)
    assert result.command == "/agent"
    assert result.mode == "deep research"


def test_detect_models_command():
    result = parse_user_input("/models")
    assert isinstance(result, ModelsCommand)
    assert result.command == "/models"
    assert result.query is None


def test_detect_models_query_command():
    result = parse_user_input(" /models opus ")
    assert isinstance(result, ModelsCommand)
    assert result.command == "/models"
    assert result.query == "opus"


def test_detect_reasoning_command():
    result = parse_user_input("/reasoning")
    assert isinstance(result, ReasoningCommand)
    assert result.command == "/reasoning"
    assert result.effort is None


def test_detect_reasoning_value_command():
    result = parse_user_input(" /reasoning high ")
    assert isinstance(result, ReasoningCommand)
    assert result.command == "/reasoning"
    assert result.effort == "high"


def test_detect_council_command():
    result = parse_user_input("/council")
    assert isinstance(result, CouncilCommand)
    assert result.command == "/council"
    assert result.prompt is None


def test_detect_council_question_command():
    result = parse_user_input(" /council summarize this thread ")
    assert isinstance(result, CouncilCommand)
    assert result.command == "/council"
    assert result.prompt == "summarize this thread"


def test_detect_multiline_council_question_command():
    result = parse_user_input(" /council\nSummarize this thread\nand list next steps ")
    assert isinstance(result, CouncilCommand)
    assert result.command == "/council"
    assert result.prompt == "Summarize this thread\nand list next steps"


def test_current_model_query_detects_models_arguments():
    assert current_model_query("/models opus") == "opus"
    assert current_model_query("/models   ") == ""
    assert current_model_query("/models") is None


def test_current_reasoning_query_detects_reasoning_arguments():
    assert current_reasoning_query("/reasoning high") == "high"
    assert current_reasoning_query("/reasoning   ") == ""
    assert current_reasoning_query("/reasoning") is None


def test_should_expand_models_query_only_for_exact_command():
    assert should_expand_models_query("/models") is True
    assert should_expand_models_query(" /models") is True
    assert should_expand_models_query("/models ") is False
    assert should_expand_models_query("/models opus") is False


def test_should_expand_reasoning_query_only_for_exact_command():
    assert should_expand_reasoning_query("/reasoning") is True
    assert should_expand_reasoning_query(" /reasoning") is True
    assert should_expand_reasoning_query("/reasoning ") is False
    assert should_expand_reasoning_query("/reasoning high") is False


def test_should_submit_save_on_enter():
    assert should_submit_on_enter("/save ") is True


def test_should_submit_quit_on_enter():
    assert should_submit_on_enter("/quit") is True


def test_should_submit_agent_on_enter():
    assert should_submit_on_enter("/agent") is True
    assert should_submit_on_enter("/agent standard") is True
    assert should_submit_on_enter("/agent conversational") is True
    assert should_submit_on_enter("/agent deep research") is True


def test_should_submit_models_on_enter():
    assert should_submit_on_enter("/models") is True
    assert should_submit_on_enter("/models opus") is True


def test_should_submit_reasoning_on_enter():
    assert should_submit_on_enter("/reasoning") is True
    assert should_submit_on_enter("/reasoning medium") is True


def test_should_submit_council_on_enter():
    assert should_submit_on_enter("/council") is False
    assert should_submit_on_enter("/council ") is False
    assert should_submit_on_enter("/council summarize this") is True
    assert should_submit_on_enter("/council\nsummarize this") is False


def test_is_council_multiline_input():
    assert is_council_multiline_input("/council\n") is True
    assert is_council_multiline_input(" /council\nquestion") is True
    assert is_council_multiline_input("/council summarize this") is False
    assert is_council_multiline_input("/council") is False
    assert is_council_multiline_input("hello\n/council") is False


def test_should_not_submit_normal_prompt_on_enter():
    assert should_submit_on_enter("hello") is False


def test_should_not_submit_multiline_text_on_enter():
    assert should_submit_on_enter("/quit\nnow") is False


def test_detect_bang_only():
    result = parse_user_input("!")
    assert isinstance(result, NormalPrompt)
    assert result.text == "!"


def test_format_execution_for_context():
    execution = BashExecution(
        command="git status",
        output="On branch main\n",
        exit_code=0,
        timestamp=datetime.now(),
        exclude_from_context=False,
    )
    text = format_for_context(execution)
    assert "Shell command:" in text
    assert "git status" in text
    assert "Output:" in text
    assert "On branch main" in text


def test_format_execution_for_display():
    execution = BashExecution(
        command="git status",
        output="On branch main\n",
        exit_code=0,
        timestamp=datetime.now(),
        exclude_from_context=False,
    )
    text = format_for_display(execution)
    assert "Command:" in text
    assert "git status" in text
    assert "Output:" in text
    assert "On branch main" in text
