import asyncio
from pathlib import Path
from  trae_agent.tools.edit_tool import TextEditorTool  # 替换为实际 import 路径

async def main():
    tool = TextEditorTool()

    args = {
        "command": "create",
        "path": "/mnt/jiaxiang/test_created_by_tool.py",  # 绝对路径
        "file_text": "# hello\nprint('created by TextEditorTool')\n"
    }

    result = await tool.execute(args)
    # ToolExecResult 通常有 .output / .error / .error_code
    if getattr(result, "error", None):
        print("TOOL ERROR:", result.error)
    else:
        print("SUCCESS:", getattr(result, "output", None))

asyncio.run(main())
