"""proactive_reply 内置工具。"""

from typing import Any, Optional

import traceback

from src.chat.replyer.replyer_manager import replyer_manager
from src.cli.maisaka_cli_sender import CLI_PLATFORM_NAME, render_cli_message
from src.common.data_models.reply_generation_data_models import ReplyGenerationResult
from src.common.logger import get_logger
from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolInvocation, ToolSpec
from src.maisaka.message_adapter import build_visible_text_from_sequence
from src.services import send_service

from .context import BuiltinToolRuntimeContext
from .reply import _run_expression_selector

logger = get_logger("maisaka_builtin_proactive_reply")


def get_tool_spec() -> ToolSpec:
    """获取 proactive_reply 工具声明。"""

    return ToolSpec(
        name="proactive_reply",
        brief_description="在自动聊天触发时，生成并发送一条不引用具体用户消息的可见发言。",
        detailed_description=(
            "仅在系统提示当前处于自动聊天/主动开口场景时使用。\n"
            "参数说明：\n"
            "- reference_info：string，可选。上文中有助于发起话题的参考信息，使用平文本格式。"
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "reference_info": {
                    "type": "string",
                    "description": "有助于主动发言的信息，之前搜集得到的事实性信息、记忆或当前话题线索。",
                    "default": "",
                },
            },
        },
        provider_name="maisaka_builtin",
        provider_type="builtin",
    )


def _build_monitor_metadata(reply_result: ReplyGenerationResult) -> dict[str, object]:
    """从 reply 结果中提取统一监控详情。"""

    monitor_detail = reply_result.monitor_detail
    if isinstance(monitor_detail, dict):
        return {"monitor_detail": monitor_detail}
    return {}


def _build_send_result(
    *,
    index: int,
    segment: str,
    success: bool,
    message_id: str = "",
) -> dict[str, Any]:
    """构建分段主动发言的轻量发送结果。"""

    return {
        "index": index,
        "segment": segment,
        "set_quote": False,
        "success": success,
        "message_id": message_id,
    }


async def handle_tool(
    tool_ctx: BuiltinToolRuntimeContext,
    invocation: ToolInvocation,
    context: Optional[ToolExecutionContext] = None,
) -> ToolExecutionResult:
    """执行 proactive_reply 内置工具。"""

    latest_thought = context.reasoning if context is not None else invocation.reasoning
    reference_info = str(invocation.arguments.get("reference_info") or "").strip()

    if not tool_ctx.runtime.is_auto_chat_turn_active():
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "proactive_reply 只能在自动聊天触发的回合中使用。",
        )

    try:
        replyer = replyer_manager.get_replyer(
            chat_stream=tool_ctx.runtime.chat_stream,
            request_type="maisaka_replyer",
            replyer_type="maisaka",
        )
    except Exception:
        logger.exception(f"{tool_ctx.runtime.log_prefix} 获取主动发言生成器时发生异常")
        logger.info(traceback.format_exc())
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "获取 Maisaka 主动发言生成器时发生异常。",
        )

    if replyer is None:
        logger.error(f"{tool_ctx.runtime.log_prefix} 获取主动发言生成器失败")
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "Maisaka 主动发言生成器当前不可用。",
        )

    replyer_chat_history = list(tool_ctx.runtime._chat_history)

    try:
        success, reply_result = await replyer.generate_reply_with_context(
            reply_reason=latest_thought,
            reference_info=reference_info,
            stream_id=tool_ctx.runtime.session_id,
            reply_message=None,
            chat_history=replyer_chat_history,
            sub_agent_runner=lambda system_prompt: _run_expression_selector(
                tool_ctx,
                system_prompt,
            ),
            log_reply=False,
        )
    except Exception as exc:
        logger.exception(f"{tool_ctx.runtime.log_prefix} 主动发言生成器执行异常: {exc}")
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "生成主动发言时发生异常。",
        )

    reply_metadata = _build_monitor_metadata(reply_result)
    reply_text = reply_result.completion.response_text.strip() if success else ""
    if not reply_text:
        logger.warning(
            f"{tool_ctx.runtime.log_prefix} 主动发言生成器返回空文本: "
            f"错误信息={reply_result.error_message!r}"
        )
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "生成主动发言失败。",
            metadata=reply_metadata,
        )

    reply_sequences = tool_ctx.post_process_reply_message_sequences(reply_text)
    reply_segments = [build_visible_text_from_sequence(sequence) for sequence in reply_sequences]
    combined_reply_text = "".join(reply_segments)
    send_results: list[dict[str, Any]] = []

    try:
        sent = False
        if tool_ctx.runtime.chat_stream.platform == CLI_PLATFORM_NAME:
            for index, segment in enumerate(reply_segments):
                render_cli_message(segment)
                send_results.append(_build_send_result(index=index, segment=segment, success=True))
            sent = True
        else:
            for index, reply_sequence in enumerate(reply_sequences):
                segment = reply_segments[index]
                sent_message = await send_service._send_to_target_with_message(
                    message_sequence=reply_sequence,
                    stream_id=tool_ctx.runtime.session_id,
                    display_message=segment,
                    set_reply=False,
                    reply_message=None,
                    selected_expressions=reply_result.selected_expression_ids or None,
                    typing=index > 0,
                    sync_to_maisaka_history=True,
                    maisaka_source_kind="guided_reply",
                )
                sent = sent_message is not None
                send_results.append(
                    _build_send_result(
                        index=index,
                        segment=segment,
                        success=sent,
                        message_id=sent_message.message_id if sent_message is not None else "",
                    )
                )
                if not sent:
                    break

        if not sent:
            return tool_ctx.build_failure_result(
                invocation.tool_name,
                "发送主动发言失败。",
                metadata=reply_metadata,
            )
    except Exception as exc:
        logger.exception(f"{tool_ctx.runtime.log_prefix} 发送主动发言时发生异常: {exc}")
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "发送主动发言时发生异常。",
            metadata=reply_metadata,
        )

    tool_ctx.runtime.mark_auto_chat_visible_output()
    metadata = {
        **reply_metadata,
        "reply_text": combined_reply_text,
        "reply_segments": reply_segments,
        "send_results": send_results,
    }
    return tool_ctx.build_success_result(
        invocation.tool_name,
        f"主动发言已发送：{combined_reply_text}",
        structured_content={
            "reply_text": combined_reply_text,
            "reply_segments": reply_segments,
            "send_results": send_results,
        },
        metadata=metadata,
    )
