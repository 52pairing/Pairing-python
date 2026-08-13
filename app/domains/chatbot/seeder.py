"""챗봇 지식 청크를 임베딩해서 저장한다.

여러 번 돌려도 안전하다. 원문 해시가 같으면 임베딩 API 를 부르지 않고 건너뛴다.
그래서 배포마다 무조건 호출해도 비용이 늘지 않는다.
"""

import hashlib
import logging

from app.clients.gemini import GeminiClient, GeminiTask
from app.domains.chatbot.knowledge import KNOWLEDGE_CHUNKS
from app.domains.chatbot.repository import ChatbotKnowledgeRepository
from app.domains.chatbot.schemas import KnowledgeReindexResponse

logger = logging.getLogger(__name__)


class ChatbotKnowledgeSeeder:
    def __init__(self, gemini: GeminiClient, repository: ChatbotKnowledgeRepository):
        self._gemini = gemini
        self._repository = repository

    async def seed(self) -> KnowledgeReindexResponse:
        model = self._gemini.model_for(GeminiTask.EMBEDDING)

        pending: list[tuple[str, str, str]] = []  # (chunk_key, content, source_hash)
        for chunk_key, content in KNOWLEDGE_CHUNKS:
            source_hash = _hash(content, model)
            if await self._repository.find_hash(chunk_key) == source_hash:
                continue
            pending.append((chunk_key, content, source_hash))

        skipped = len(KNOWLEDGE_CHUNKS) - len(pending)
        if not pending:
            logger.info("[챗봇 지식 시딩] 변경 없음 (%d건 유지)", skipped)
            return KnowledgeReindexResponse(total=len(KNOWLEDGE_CHUNKS), embedded=0, skipped=skipped)

        # 한 번에 임베딩한다. 청크마다 호출하면 왕복이 청크 수만큼 늘어난다.
        vectors = await self._gemini.embed([content for _, content, _ in pending])

        for (chunk_key, content, source_hash), vector in zip(pending, vectors, strict=True):
            await self._repository.upsert(chunk_key, content, vector, model, source_hash)

        logger.info("[챗봇 지식 시딩] 신규·변경 %d건, 유지 %d건", len(pending), skipped)
        return KnowledgeReindexResponse(total=len(KNOWLEDGE_CHUNKS), embedded=len(pending), skipped=skipped)


def _hash(content: str, model: str) -> str:
    """모델명을 같이 넣는다. 모델이 바뀌면 차원이 같아도 벡터 공간이 달라서 재생성해야 한다."""
    return hashlib.sha256(f"{model}\n{content}".encode()).hexdigest()
