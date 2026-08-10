-- =====================================================================
-- AI 서버 소유 테이블  [PostgreSQL + pgvector]
--
-- 선행: 백엔드 레포의 db/init/02-create-schema.sql
-- 실행: psql -U pairing -d pairing -v ON_ERROR_STOP=1 -f ai-server/db/init/10-create-ai-schema.sql
--
-- 소유권
--   여기 있는 테이블만 AI 서버가 INSERT/UPDATE 한다.
--   스프링 소유 테이블(account, project_position 등)은 읽기만 하고,
--   상태를 바꿔야 하면 스프링 API 를 호출한다.
--
-- 차원은 768로 고정한다(현재 모델: gemini-embedding-001, output_dimensionality=768로 축소 지정).
-- 모델을 바꿔도 output_dimensionality를 768로 맞추면 이 컬럼은 안 바꿔도 된다. 차원 자체를
-- 바꾸려면 이 컬럼과 app.core.config 의 embedding_dimension 을 함께 바꾼다.
-- 주의: 모델을 바꾸면 차원이 같아도 벡터 공간 자체가 달라진다 — 기존 벡터와 섞이면 안 되므로
-- Pairing-backend의 POST /api/v1/matchings/admin/embeddings/reindex 로 기존 벡터를 재생성해야 한다.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;


CREATE TABLE IF NOT EXISTS "freelancer_embedding" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY NOT NULL,
    "freelancer_id" BIGINT NOT NULL,
    "embedding" vector(768) NOT NULL,
    "model" VARCHAR(50) NOT NULL,
    "source_hash" VARCHAR(64) NOT NULL,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "position_embedding" (
    "id" BIGINT GENERATED ALWAYS AS IDENTITY NOT NULL,
    "position_id" BIGINT NOT NULL,
    "embedding" vector(768) NOT NULL,
    "model" VARCHAR(50) NOT NULL,
    "source_hash" VARCHAR(64) NOT NULL,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    PRIMARY KEY ("id")
);


-- 대상당 한 행만 유지한다. (upsert 기준)
ALTER TABLE "freelancer_embedding" ADD CONSTRAINT "uk_freelancer_embedding" UNIQUE ("freelancer_id");
ALTER TABLE "position_embedding" ADD CONSTRAINT "uk_position_embedding" UNIQUE ("position_id");

-- FK 는 걸지 않는다. 스프링이 프로필을 지울 때 AI 테이블 때문에 삭제가 막히면 안 된다.
-- 고아 행은 배치로 정리한다.

-- 코사인 거리 인덱스. 행이 수천 건 미만이면 순차 스캔이 더 빠를 수 있어,
-- 데이터가 쌓인 뒤 EXPLAIN 으로 확인하고 lists 값을 조정한다.
CREATE INDEX IF NOT EXISTS "idx_freelancer_embedding_cosine"
    ON "freelancer_embedding" USING ivfflat ("embedding" vector_cosine_ops) WITH (lists = 100);

COMMENT ON TABLE "freelancer_embedding" IS '프리랜서 이력서/조건 임베딩 (AI 서버 소유)';
COMMENT ON TABLE "position_embedding" IS '프로젝트 포지션 요구조건 임베딩 (AI 서버 소유)';
COMMENT ON COLUMN "freelancer_embedding"."freelancer_id" IS 'freelancer_profile.id (FK 미설정)';
COMMENT ON COLUMN "freelancer_embedding"."source_hash" IS '원문 해시. 같으면 재생성하지 않는다';
COMMENT ON COLUMN "freelancer_embedding"."model" IS '생성에 쓴 임베딩 모델명';
