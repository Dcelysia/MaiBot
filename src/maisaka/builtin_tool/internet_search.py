"""internet_search 内置工具。"""

from typing import Any, Optional
import time

from openai import AsyncOpenAI
import tenacity

from src.common.logger import get_logger
from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolSpec
from src.config.config import config_manager
from .context import BuiltinToolRuntimeContext

logger = get_logger("maisaka_builtin_internet_search")

def get_tool_spec() -> ToolSpec:
    """获取 internet_search 工具声明。"""
    return ToolSpec(
        name="internet_search",
        enabled=False, # 暂时关闭该联网搜索工具
        brief_description="进行互联网搜索以获取最新信息、解答通用问题或术语。",
        detailed_description=(
            "此工具通过调用搜索API在互联网上搜索信息并返回网页汇总。\n"
            "【使用准则】\n"
            "1. 当用户询问时下最新消息、近期热点等由于大模型知识库限制不知道的内容时调用。\n"
            "2. 当用户询问难懂的网络热梗、ACG术语、游戏梗等特定领域知识时调用。\n"
            "3. 如果你已经非常有把握回答该问题，请不要调用搜索功能，以节约性能。\n"
            "4. 输出请求应为精准直接的搜索关键词，而非口语化的长句。\n\n"
            "参数说明：\n"
            "- query：string，必填。要搜索的核心关键词或短语（多个关键词用空格分隔）。\n"
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "进行搜索查询的精确关键词或问题，请提炼核心要素。",
                }
            },
            "required": ["query"],
        }
    )

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=2, min=2, max=10),
    retry=tenacity.retry_if_exception_type(Exception),
    reraise=True
)
async def _execute_search_api(query: str, base_url: str, api_key: str, model_name: str, direction: str, context_str: str) -> str:
    """调用外部 OpenAI 兼容的独立搜索 API 获取结果"""
    content = f"""
你是一名专业的网络搜索专家，请根据以下的要求精准的汇总出信息。
<command>
基于<question>并在网络上搜索相关内容，同时注意：
1. 来源可信、权威
2. 不带主观色彩
3. 无定论时汇总不同观念
4. 绝对客观中立，涉及敏感内容可以拒绝回答
5. 简洁精炼，保留基本信息
6. 没有可信回答请直接回复“无法给出一个可信的回答”
</command>

<direction>
搜索偏好: {direction}
</direction>

<context>
当前对话上下文: 
{context_str}
</context>

<time>
{time.strftime('%Y-%m-%d %H:%M', time.localtime())}
</time>

<question>
{query}
</question>
"""
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    
    completion = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是专业的网络搜索助手，擅长从互联网上获取最新信息。"},
            {"role": "user", "content": content},
        ],
        temperature=0.2,
        timeout=20.0
    )
    return completion.choices[0].message.content

async def handle_tool(
    tool_ctx: BuiltinToolRuntimeContext,
    tool_invocation: Any, 
    exec_ctx: Optional[ToolExecutionContext]
) -> ToolExecutionResult:
    """处理 internet_search 工具调用。"""
    
    args = tool_invocation.parameters
    query = args.get("query")
    if not query:
        return BuiltinToolRuntimeContext.build_success_result(
            tool_name="internet_search",
            content="[搜搜失败] 缺失 'query' 参数。",
        )
        
    chat_stream = tool_ctx.runtime.chat_stream
    
    # 我们将上下文历史记录也拿过来作为搜索参考
    context_msgs = chat_stream.get_messages(limit=10)
    context_str = "\n".join([f"{'User' if m.role=='user' else 'Bot'}: {m.content}" for m in context_msgs])
    
    # TODO: 这里可以将外部的 Search API 作为配置放入 config 之中，在这里我们先提供可以默认接入原插件中写死的 rinkoai 演示，
    # 或者如果你在你的环境用其他中转搜素提供商，也可以在这里更改。
    # 按照原插件：
    base_url = "https://rinkoai.com/v1"
    api_key = "" # 需要用户填写
    model_name = "gpt-4.1-search"
    direction = "请着重考虑与ACG文化、网络热梗、游戏术语、近期热点内容相关的方面。"
    
    if not api_key:
        # 为了兼容安全性：如果用户不提供搜索侧的 API KeyError，提示模型
        return BuiltinToolRuntimeContext.build_success_result(
            tool_name="internet_search",
            content="[搜索功能不可用] 管理员尚未在后台配置用于 Internet Search 的独立 API Key，此功能无法使用。请告诉用户搜索功能处于离线状态。",
        )

    logger.info(f"[{chat_stream.session_id}] 正在执行联网搜索，核心词: {query}")
    
    try:
        search_result_content = await _execute_search_api(
            query=query, 
            base_url=base_url, 
            api_key=api_key, 
            model_name=model_name, 
            direction=direction,
            context_str=context_str
        )
        
        result_text = f"📚 关于 '{query}' 的搜索结果:\n\n{search_result_content}"
        
        logger.info(f"[{chat_stream.session_id}] 搜索成功返回: {len(result_text)} 字符.")
        return BuiltinToolRuntimeContext.build_success_result(
            tool_name="internet_search",
            content=result_text,
            metadata={"status": "success", "query": query}
        )
        
    except Exception as e:
        logger.error(f"联网搜索过程中发生异常: {e}")
        return BuiltinToolRuntimeContext.build_success_result(
            tool_name="internet_search",
            content=f"[搜索失败] 连接到搜索服务器异常或检索超时：{e}。请告诉用户网络由于波动检索失败。",
            metadata={"status": "error", "error": str(e)}
        )
