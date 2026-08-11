#!/usr/bin/env python3
"""
OCR MCP Server - macOS Vision-based OCR for Claude Code.

Provides an `ocr_image` tool that extracts text from images using
macOS's built-in Vision framework (中文/English/日本語/한국어 supported).
No network, no API key needed.
"""

import json
import subprocess
import sys
import os
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

OCR_TOOL_PATH = Path(__file__).parent / "ocr_tool"

server = Server("ocr-mcp-server")


@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="ocr_image",
            description="Extract text from an image file using macOS Vision OCR. "
            "Supports Chinese (Simplified & Traditional), English, Japanese, Korean. "
            "Accepts a local image file path and returns the recognized text. "
            "Use this whenever the user provides an image/screenshot and you need to read its text content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to the image file (PNG, JPEG, TIFF, etc.)",
                    }
                },
                "required": ["image_path"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name != "ocr_image":
        raise ValueError(f"Unknown tool: {name}")

    image_path = arguments.get("image_path", "")

    if not image_path:
        return [TextContent(type="text", text="Error: image_path is required")]

    path = Path(image_path)
    if not path.exists():
        return [TextContent(
            type="text",
            text=f"Error: Image file not found: {image_path}",
        )]

    # Check if it looks like an image
    suffix = path.suffix.lower()
    valid_suffixes = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".heic", ".heif"}
    if suffix not in valid_suffixes:
        return [TextContent(
            type="text",
            text=f"Error: Unsupported image format: {suffix}. Supported: {', '.join(sorted(valid_suffixes))}",
        )]

    try:
        result = subprocess.run(
            [str(OCR_TOOL_PATH), str(path.resolve())],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return [TextContent(
                type="text",
                text=f"OCR error (exit code {result.returncode}): {result.stderr.strip()}",
            )]

        text = result.stdout.strip()
        if not text:
            return [TextContent(
                type="text",
                text="OCR completed, but no text was found in the image.",
            )]

        return [TextContent(
            type="text",
            text=f"OCR result from '{path.name}':\n\n{text}",
        )]

    except subprocess.TimeoutExpired:
        return [TextContent(type="text", text="Error: OCR timed out after 30 seconds")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error running OCR: {str(e)}")]


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
