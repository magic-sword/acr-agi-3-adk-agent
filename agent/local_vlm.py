"""Google ADK model connector for an offline, OpenAI-compatible vision server.

The server runs on the Docker network during local evaluation and inside the
Kaggle notebook during the competition rerun. Only text and PNG image inputs
are needed by this agent, so this connector has no extra Python dependencies.
"""
from __future__ import annotations

import asyncio
import base64
import json
import urllib.request
from typing import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


class LocalVisionLlm(BaseLlm):
    api_base: str
    timeout_seconds: int = 180
    max_output_tokens: int = 96

    def _complete(self, messages: list[dict]) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.api_base.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            result = json.load(response)
        answer = result["choices"][0]["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Local vision model returned no text: {result!r}")
        return answer

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        if llm_request.tools_dict or llm_request.config.tools:
            raise ValueError("The local vision model adapter does not support tool calls")
        messages: list[dict] = []
        instruction = llm_request.config.system_instruction
        if instruction:
            if isinstance(instruction, types.Content):
                instruction = "\n".join(p.text for p in instruction.parts or [] if p.text)
            messages.append({"role": "system", "content": str(instruction)})
        for content in llm_request.contents:
            parts: list[dict] = []
            for part in content.parts or []:
                if part.text:
                    parts.append({"type": "text", "text": part.text})
                elif part.inline_data:
                    blob = part.inline_data
                    if not blob.mime_type.startswith("image/"):
                        raise ValueError(f"Unsupported model input: {blob.mime_type}")
                    encoded = base64.b64encode(blob.data).decode("ascii")
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{blob.mime_type};base64,{encoded}"},
                    })
                else:
                    raise ValueError("Unsupported ADK part in local vision model input")
            if parts:
                messages.append({
                    "role": "assistant" if content.role == "model" else "user",
                    "content": parts,
                })
        if not messages or all(m["role"] == "system" for m in messages):
            raise ValueError("The vision model needs an observation")
        answer = await asyncio.to_thread(self._complete, messages)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=answer)]))
