"""mute_user 内置工具。"""

from typing import Any, Optional
import httpx

from src.common.logger import get_logger
from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolSpec
from .context import BuiltinToolRuntimeContext

logger = get_logger("maisaka_builtin_mute_user")

def get_tool_spec() -> ToolSpec:
    """获取 mute_user 工具声明。"""
    return ToolSpec(
        name="mute_user",
        brief_description="对群内违规用户执行禁言或解除禁言操作。",
        detailed_description=(
            "此工具是对群成员进行群禁言的物理操作接口。\n"
            "【绝对使用准则】\n"
            "1. 当用户出现严重扰乱群聊秩序的行为（如刷屏、恶意辱骂、挑衅、违规内容）时主动使用。\n"
            "2. 当群主或高级管理员明确要求你禁言某人时使用。\n"
            "3. 当认为可以作为一种傲娇反击或生气的人设互动（短暂禁言作为警告，如60秒）时适度使用。\n"
            "4. 绝对不可滥用！不可无故禁言正常聊天的群员。\n\n"
            "参数说明：\n"
            "- user_id：string，必填。需要禁言或解禁的目标用户的数字ID（如QQ号）。仔细思考不要弄错对象。\n"
            "- duration：integer，必填。禁言时间长度，单位秒。0 代表解除禁言。最大范围 2592000 秒。\n"
            "- reason：string，必填。解释进行此次禁言/解禁操作的充分理由，该理由将记录在系统日志中。\n"
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "目标群用户的数字ID（如QQ号）。",
                },
                "duration": {
                    "type": "integer",
                    "description": "禁言时间长度，单位秒。0 代表解除禁言。",
                },
                "reason": {
                    "type": "string",
                    "description": "必须说明执行该操作的原因和理由。",
                }
            },
            "required": ["user_id", "duration", "reason"],
        }
    )

async def handle_tool(
    tool_ctx: BuiltinToolRuntimeContext,
    tool_invocation: Any, 
    exec_ctx: Optional[ToolExecutionContext]
) -> ToolExecutionResult:
    """处理 mute_user 工具调用。"""
    
    args = tool_invocation.parameters
    user_id = args.get("user_id")
    duration = int(args.get("duration", 0))
    reason = args.get("reason", "未提供理由")
    
    chat_stream = tool_ctx.runtime.chat_stream
    if not chat_stream.is_group_session:
        return BuiltinToolRuntimeContext.build_success_result(
            tool_name="mute_user",
            content="[操作失败] 此工具只能在群聊环境中使用。",
            metadata={"status": "failed", "reason": "not_in_group"}
        )
        
    group_id = getattr(chat_stream, "group_id", None)

    if not group_id:
        return BuiltinToolRuntimeContext.build_success_result(
            tool_name="mute_user",
            content="[操作失败] 无法获取当前群聊的 group_id，可能不支持此平台。",
            metadata={"status": "failed", "reason": "missing_group_id"}
        )
    
    # 我们假设平台主要是 OneBot v11，使用通过 httpx 呼叫 set_group_ban 的方式
    napcat_api = "http://127.0.0.1:3000/set_group_ban"
    payload = {
        "group_id": str(group_id),
        "user_id": str(user_id),
        "duration": duration
    }
    
    action_str = "禁言" if duration > 0 else "解除禁言"
    logger.info(f"[{chat_stream.session_id}] 尝试给用户 {user_id} 发送 {action_str} 请求, 持续 {duration} 秒. 理由: {reason}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(napcat_api, json=payload, timeout=5)
            if response.status_code == 200:
                resp_json = response.json()
                if resp_json.get("status") == "ok" and resp_json.get("retcode") == 0:
                    logger.info(f"[{chat_stream.session_id}] {action_str} {user_id} 成功.")
                    return BuiltinToolRuntimeContext.build_success_result(
                        tool_name="mute_user",
                        content=f"[操作成功] 已成功对目标用户执行了 {action_str} 操作，时长 {duration} 秒。理由：{reason}",
                        metadata={"status": "success", "duration": duration, "target": user_id}
                    )
                else:
                    logger.error(f"Napcat 禁言失败返回: {resp_json}")
                    return BuiltinToolRuntimeContext.build_success_result(
                        tool_name="mute_user",
                        content=f"[操作失败] 适配器执行拒绝或失败: {resp_json.get('msg', resp_json)}",
                        metadata={"status": "failed", "response": resp_json}
                    )
            else:
                logger.error(f"Napcat 禁言请求HTTP失败: {response.status_code}")
                return BuiltinToolRuntimeContext.build_success_result(
                    tool_name="mute_user",
                    content=f"[操作失败] 请求适配器状态码错误: {response.status_code}",
                    metadata={"status": "failed", "status_code": response.status_code}
                )
    except Exception as e:
        logger.error(f"Napcat 禁言异常: {e}")
        return BuiltinToolRuntimeContext.build_success_result(
            tool_name="mute_user",
            content=f"[操作失败] 内部接口连接发送异常: {e}",
            metadata={"status": "error", "error": str(e)}
        )
