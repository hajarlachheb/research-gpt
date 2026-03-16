"""LLM answer generation with citation grounding and streaming support."""

from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from config.settings import get_settings
from prompts.manager import PromptManager
from monitoring.logger import get_logger

logger = get_logger(__name__)


class LLMGenerator:
    def __init__(self, prompt_manager: PromptManager | None = None):
        self.settings = get_settings()
        self._client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self._prompt_manager = prompt_manager or PromptManager()

    async def generate(self, question: str, context: str) -> tuple[str, dict]:
        """Generate a grounded answer. Returns (answer_text, usage_dict)."""
        system_prompt = self._prompt_manager.get_prompt("answer")
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

        logger.info("llm_generate_start", question_len=len(question), context_len=len(context))

        response = await self._client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=self.settings.max_response_tokens,
        )

        answer = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        logger.info("llm_generate_complete", usage=usage)
        return answer, usage

    async def generate_stream(self, question: str, context: str) -> AsyncIterator[str]:
        """Stream answer tokens as they are generated."""
        system_prompt = self._prompt_manager.get_prompt("answer")
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

        logger.info("llm_stream_start", question_len=len(question))

        stream = await self._client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=self.settings.max_response_tokens,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
