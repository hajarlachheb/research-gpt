"""Rewrite user queries to improve retrieval quality."""

from __future__ import annotations

from openai import AsyncOpenAI

from config.settings import get_settings
from prompts.manager import PromptManager
from monitoring.logger import get_logger

logger = get_logger(__name__)


class QueryRewriter:
    def __init__(self, prompt_manager: PromptManager | None = None):
        self.settings = get_settings()
        self._client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self._prompt_manager = prompt_manager or PromptManager()

    async def rewrite(self, query: str) -> str:
        """Rewrite a user query for better retrieval against research papers."""
        system_prompt = self._prompt_manager.get_prompt("rewrite")

        try:
            response = await self._client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            rewritten = response.choices[0].message.content or query
            rewritten = rewritten.strip().strip('"').strip("'")
            logger.info("query_rewritten", original=query, rewritten=rewritten)
            return rewritten
        except Exception as exc:
            logger.warning("query_rewrite_failed", error=str(exc))
            return query
