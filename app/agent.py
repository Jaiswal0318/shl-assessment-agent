"""Core agent: retrieval, Gemini LLM calls, and response validation."""

from __future__ import annotations

import asyncio
import json
import logging
import re

import google.generativeai as genai

from app.catalog import Assessment, build_name_index, build_url_index
from app.config import GEMINI_API_KEY, GEMINI_MODEL, MAX_TURNS, TOP_K_RECOMMEND, TOP_K_RETRIEVAL
from app.conversation import count_turns, get_last_user_message, is_off_topic
from app.models import ChatResponse, Recommendation
from app.prompts import RETRIEVAL_CONTEXT_TEMPLATE, SYSTEM_PROMPT, format_retrieved_assessments
from app.retrieval import RetrievalEngine

logger = logging.getLogger(__name__)

REFUSAL_REPLY = (
    "I'm sorry, I can only help with SHL assessment recommendations. "
    "Could you tell me about the role you're hiring for?"
)


class SHLAgent:
    """Orchestrates retrieval, Gemini generation, and catalog-grounded validation."""

    def __init__(self, assessments: list[Assessment]):
        self.assessments = assessments
        self.url_index = build_url_index(assessments)
        self.name_index = build_name_index(assessments)
        self.retrieval = RetrievalEngine(assessments)

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=2048,
                response_mime_type="application/json",
            ),
        )
        logger.info("Agent initialized with %d assessments | Model: %s", len(assessments), GEMINI_MODEL)

    async def process_chat(self, messages: list[dict]) -> ChatResponse:
        try:
            turn_count = count_turns(messages)
            last_user = get_last_user_message(messages)

            if is_off_topic(last_user):
                return ChatResponse(reply=REFUSAL_REPLY, recommendations=[], end_of_conversation=False)

            search_query = self._build_search_query(messages)
            retrieved = self.retrieval.search(search_query, top_k=TOP_K_RETRIEVAL)
            retrieved_context = format_retrieved_assessments(retrieved)
            retrieval_prompt = RETRIEVAL_CONTEXT_TEMPLATE.format(retrieved_assessments=retrieved_context)

            prompt = self._build_prompt(messages, retrieval_prompt, turn_count)
            raw_response = await self._call_gemini(prompt)
            return self._parse_and_validate(raw_response)

        except Exception as exc:
            logger.error("Agent error: %s", exc, exc_info=True)
            return ChatResponse(
                reply=(
                    "I apologize, but I encountered an issue processing your request. "
                    "Could you please rephrase your question about SHL assessments?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

    def _build_search_query(self, messages: list[dict]) -> str:
        user_parts = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
        return " ".join(user_parts)

    def _build_prompt(self, messages: list[dict], retrieval_context: str, turn_count: int) -> str:
        lines: list[str] = ["CONVERSATION HISTORY:"]
        for msg in messages:
            role = msg.get("role", "user").upper()
            lines.append(f"{role}: {msg.get('content', '')}")

        turn_budget = MAX_TURNS - turn_count
        if turn_budget <= 2:
            budget_note = f"Turns used: {turn_count}/{MAX_TURNS}. MUST recommend NOW."
        elif turn_budget <= 4:
            budget_note = f"Turns used: {turn_count}/{MAX_TURNS}. Recommend soon if enough context."
        else:
            budget_note = f"Turns used: {turn_count}/{MAX_TURNS}. Clarify if needed."

        lines.append(f"\n{retrieval_context}")
        lines.append(f"\n[{budget_note}]")
        lines.append("\nRespond with valid JSON only.")
        return "\n".join(lines)

    async def _call_gemini(self, prompt: str) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(self.model.generate_content, prompt)
                return response.text or ""
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = 2**attempt
                    logger.warning("Gemini API error (attempt %d), retrying in %ds: %s", attempt + 1, wait, exc)
                    await asyncio.sleep(wait)
                else:
                    raise

    def _parse_and_validate(self, raw_response: str) -> ChatResponse:
        text = raw_response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)

        try:
            data = json.loads(text, strict=False)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    data = json.loads(match.group(), strict=False)
                except json.JSONDecodeError:
                    return self._fallback_response(text)
            else:
                return self._fallback_response(text)

        reply = data.get("reply", "")
        recommendations_raw = data.get("recommendations", [])
        end_of_conversation = bool(data.get("end_of_conversation", False))

        valid_recommendations: list[Recommendation] = []
        if recommendations_raw and isinstance(recommendations_raw, list):
            for rec in recommendations_raw:
                if not isinstance(rec, dict):
                    continue

                name = rec.get("name", "")
                url = rec.get("url", "")

                if url in self.url_index:
                    canonical = self.url_index[url]
                    valid_recommendations.append(
                        Recommendation(
                            name=canonical.name,
                            url=canonical.url,
                            test_type=canonical.test_type_codes,
                        )
                    )
                elif name.lower() in self.name_index:
                    canonical = self.name_index[name.lower()]
                    valid_recommendations.append(
                        Recommendation(
                            name=canonical.name,
                            url=canonical.url,
                            test_type=canonical.test_type_codes,
                        )
                    )
                else:
                    matched = self._fuzzy_match_assessment(name)
                    if matched:
                        valid_recommendations.append(
                            Recommendation(
                                name=matched.name,
                                url=matched.url,
                                test_type=matched.test_type_codes,
                            )
                        )
                    else:
                        logger.warning("Dropping hallucinated recommendation: %s (%s)", name, url)

        seen_urls: set[str] = set()
        deduped: list[Recommendation] = []
        for rec in valid_recommendations:
            if rec.url not in seen_urls:
                seen_urls.add(rec.url)
                deduped.append(rec)

        return ChatResponse(
            reply=reply,
            recommendations=deduped[:TOP_K_RECOMMEND],
            end_of_conversation=end_of_conversation,
        )

    def _fallback_response(self, text: str) -> ChatResponse:
        return ChatResponse(
            reply=text[:500]
            if text
            else "I can help you find the right SHL assessments. What role are you hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    def _fuzzy_match_assessment(self, name: str) -> Assessment | None:
        name_lower = name.lower().strip()
        for catalog_name, assessment in self.name_index.items():
            if name_lower in catalog_name or catalog_name in name_lower:
                return assessment

        name_words = set(name_lower.split())
        best_match: Assessment | None = None
        best_overlap = 0
        for catalog_name, assessment in self.name_index.items():
            overlap = len(name_words & set(catalog_name.split()))
            if overlap > best_overlap and overlap >= 2:
                best_overlap = overlap
                best_match = assessment
        return best_match
