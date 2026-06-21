"""MCP Server entry point.

Supports two transport modes:
  - stdio (default): for local MCP clients like Claude Desktop
  - sse: for Docker / remote access over HTTP

Usage:
  smart-clip-mcp                  # stdio mode
  smart-clip-mcp --transport sse  # SSE mode on port 8000
  smart-clip-mcp --transport sse --port 9000 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import os

from fastmcp import FastMCP

from smart_clip.tools.smart_clip import smart_clip_tool
from smart_clip.tools.repurpose import repurpose_tool
from smart_clip.tools.highlight_reel import highlight_reel_tool
from smart_clip.tools.analyze_content import analyze_content_tool
from smart_clip.tools.get_edit_plan import get_edit_plan_tool

mcp = FastMCP(
    "smart-clip",
    version="0.1.0",
    instructions="AI-powered smart video clipping MCP server. "
    "Identifies highlight moments from long videos using subtitle analysis + LLM, "
    "then clips them into short-form content.",
)

# Register tools
mcp.tool(smart_clip_tool)
mcp.tool(repurpose_tool)
mcp.tool(highlight_reel_tool)
mcp.tool(analyze_content_tool)
mcp.tool(get_edit_plan_tool)


def main():
    """Run the MCP server."""
    parser = argparse.ArgumentParser(description="Smart Clip MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default=os.getenv("SMART_CLIP_TRANSPORT", "stdio"),
        help="Transport mode: stdio (default) or sse (HTTP)",
    )
    parser.add_argument("--host", default=os.getenv("SMART_CLIP_HOST", "0.0.0.0"), help="SSE host")
    parser.add_argument("--port", type=int, default=int(os.getenv("SMART_CLIP_PORT", "8000")), help="SSE port")
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
