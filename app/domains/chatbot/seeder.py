"""챗봇 지식 청크를 임베딩해서 저장한다.

여러 번 돌려도 안전하다. 원문 해시가 같으면 임베딩 API 를 부르지 않고 건너뛴다.
그래서 배포마다 무조건 호출해도 비용이 늘지 않는다.
"""

import hashlib
import logging
from collections import Counter

from app.clients.gemini import GeminiClient, GeminiTask
from app.domains.chatbot.knowledge import (
    GREETING_CHUNKS,
    KNOWLEDGE_CHUNKS,
    NEGATIVE_CHUNKS,
)
from app.domains.chatbot.repository import ChatbotKnowledgeRepository
from app.domains.chatbot.schemas import KnowledgeReindexResponse

logger = logging.getLogger(__name__)


class ChatbotKnowledgeSeeder:
    def __init__(self, gemini: GeminiClient, repository: ChatbotKnowledgeRepository):
        self._gemini = gemini
        self._repository = repository

    async def seed(self) -> KnowledgeReindexResponse:
        model = self._gemini.model_for(GeminiTask.EMBEDDING)

        # 양성(우리 서비스)과 음성(차단 본보기)을 같은 표에 함께 넣는다. 판정은 "어느 쪽에
        # 더 가까운가"이므로 둘이 같은 벡터 공간에 있어야 한다.
        all_chunks = [(key, content, "POSITIVE") for key, content in KNOWLEDGE_CHUNKS]
        all_chunks += [(key, content, "NEGATIVE") for key, content in NEGATIVE_CHUNKS]
        all_chunks += [(key, content, "GREETING") for key, content in GREETING_CHUNKS]
        total = len(all_chunks)

        pending: list[tuple[str, str, str, str]] = []  # (key, content, source_hash, polarity)
        for chunk_key, content, polarity in all_chunks:
            # polarity 를 해시에 넣는다. 문구가 같은데 극성만 뒤집는 수정이 반영되지 않으면
            # 판정이 조용히 반대로 돈다.
            source_hash = _hash(content, model, polarity)
            if await self._repository.find_hash(chunk_key) == source_hash:
                continue
            pending.append((chunk_key, content, source_hash, polarity))

        skipped = total - len(pending)
        if not pending:
            logger.info("[챗봇 지식 시딩] 변경 없음 (%d건 유지)", skipped)
            return KnowledgeReindexResponse(total=total, embedded=0, skipped=skipped)

        # 한 번에 임베딩한다. 청크마다 호출하면 왕복이 청크 수만큼 늘어난다.
        vectors = await self._gemini.embed([content for _, content, _, _ in pending])

        for (chunk_key, content, source_hash, polarity), vector in zip(
            pending, vectors, strict=True
        ):
            await self._repository.upsert(chunk_key, content, vector, model, source_hash, polarity)

        counted = Counter(polarity for _, _, _, polarity in pending)
        logger.info(
            "[챗봇 지식 시딩] 신규·변경 %d건(양성 %d, 음성 %d, 인사 %d), 유지 %d건",
            len(pending),
            counted["POSITIVE"],
            counted["NEGATIVE"],
            counted["GREETING"],
            skipped,
        )
        return KnowledgeReindexResponse(total=total, embedded=len(pending), skipped=skipped)


def _hash(content: str, model: str, polarity: str) -> str:
    """모델명을 같이 넣는다. 모델이 바뀌면 차원이 같아도 벡터 공간이 달라서 재생성해야 한다."""
    return hashlib.sha256(f"{model}\n{polarity}\n{content}".encode()).hexdigest()
