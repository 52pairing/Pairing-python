# Pairing AI Server

페어링 백엔드(Spring Boot)의 **내부 AI 서비스**입니다. 브라우저가 직접 호출하지 않고 스프링만 호출합니다.

- 임베딩 생성/저장 (같은 PostgreSQL + pgvector)
- 매칭 후보 추천 (벡터 검색 → LLM 재랭킹)
- Gemini는 **키 하나를 공유하고 용도별로 모델만 바꿉니다**

이 폴더는 자립적입니다. 별도 레포로 분리할 때 폴더째 옮기면 됩니다.

---

## 1. 빠르게 띄우기

```bash
cd ai-server
python -m venv .venv && source .venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # AI_DB_URL / INTERNAL_API_KEY / GEMINI_API_KEY 채우기
```

벡터 테이블을 만듭니다. (백엔드의 `02-create-schema.sql` 다음에 실행)

```bash
psql -U pairing -d pairing -v ON_ERROR_STOP=1 -f db/init/10-create-ai-schema.sql
```

```bash
uvicorn app.main:app --reload --port 8000
```

- Swagger: http://localhost:8000/docs
- 헬스체크: http://localhost:8000/health

```bash
pytest        # 계약 테스트 (DB·Gemini 없이 돈다)
ruff check .
```

이미 있는 conda 환경을 쓴다면 (예: `pystudy_env`):

```bash
conda activate pystudy_env
pip install pgvector asyncpg pytest pytest-asyncio ruff   # 나머지는 대개 이미 있다
python -m pytest -q
python -m uvicorn app.main:app --reload --port 8000
```

---

## 2. 구조

```text
ai-server/
├── app/
│   ├── main.py                  앱 조립 (미들웨어, 예외 핸들러, 라우터)
│   ├── api/v1/__init__.py       라우터 조립 + 내부 인증 일괄 적용
│   ├── core/
│   │   ├── config.py            환경설정. 없으면 기동 실패
│   │   ├── context.py           traceId (스프링에서 이어받음)
│   │   ├── logging.py           모든 로그에 traceId 부착
│   │   ├── middleware.py        traceId 미들웨어 + 요청 로깅
│   │   ├── response.py          ApiResponse / ErrorResponse (스프링과 동일 형식)
│   │   ├── errors.py            AiErrorCode + 전역 예외 핸들러
│   │   └── security.py          내부 호출 키 검증
│   ├── db/session.py            async 엔진/세션, Base
│   ├── clients/
│   │   ├── gemini.py            용도별 모델 선택 + 재시도
│   │   └── spring.py            스프링 역호출 (traceId 전파)
│   └── domains/                 도메인별 수직 분할
│       ├── health/router.py
│       ├── embedding/           router → service → repository → models/schemas
│       └── matching/            router → service → schemas
├── db/init/10-create-ai-schema.sql   AI 소유 테이블 (pgvector)
├── tests/
├── pyproject.toml
├── Dockerfile
└── .env.example
```

### 레이어 책임 (스프링과 대응)

| FastAPI | Spring Boot | 하는 일 |
| --- | --- | --- |
| `domains/*/router.py` | `presentation/api/*Controller` | HTTP 계약, 검증, 응답 조립 |
| `domains/*/schemas.py` | `presentation/api/request`·`response` | 요청/응답 모델(Pydantic) |
| `domains/*/service.py` | `application/service` | 유스케이스, 비즈니스 판단 |
| `domains/*/repository.py` | `domain/repository` + `infrastructure/persistence` | SQL. 이 밖으로 SQL이 새지 않음 |
| `domains/*/models.py` | `infrastructure/persistence/*JpaEntity` | 테이블 매핑 |
| `clients/` | `infrastructure` 어댑터 | 외부 호출(Gemini, 스프링) |
| `core/errors.py` | `global/exception` | 에러코드와 공통 예외 처리 |

**규칙: 라우터에 비즈니스 로직을 두지 않고, 서비스에서 SQL을 직접 쓰지 않습니다.** 스프링 쪽 규칙과 같아서 팀원이 두 레포를 오갈 때 헷갈리지 않습니다.

---

## 3. 스프링 ↔ AI 서버 통신 규약

이 절이 두 팀의 계약입니다. 바꾸려면 양쪽에 알리고 이 문서를 함께 고칩니다.

### 3-1. 인증: 공유 키

스프링은 모든 호출에 헤더를 붙입니다. 사용자 JWT는 넘기지 않습니다 — 사용자 인증은 스프링이 이미 끝냈고, 여기는 서버 대 서버 호출입니다.

```
X-Internal-Api-Key: <INTERNAL_API_KEY>
```

`/health`, `/health/ready`만 키 없이 열려 있습니다. **이 서버는 외부에 노출하지 않습니다.** 키는 방어선 하나일 뿐이고 네트워크 차단이 먼저입니다.

### 3-2. traceId 전파

스프링의 `TraceIdFilter`가 만든 값을 그대로 이어씁니다.

```
X-Trace-Id: a1b2c3d4
```

없으면 AI 서버가 새로 만들고, 응답 헤더로 돌려줍니다. 두 서버 로그를 같은 ID로 검색할 수 있어야 장애 추적이 됩니다.

### 3-3. 응답 형식 (스프링과 동일)

```jsonc
// 성공
{ "timestamp": "...", "status": 200, "code": "CANDIDATES_FOUND", "message": "...", "data": { } }

// 실패
{ "timestamp": "...", "status": 502, "errorCode": "AI_012", "message": "...", "traceId": "a1b2c3d4" }
```

프론트와 스프링이 두 서버의 응답을 **같은 파서로** 처리할 수 있게 맞춰 둔 것입니다. 필드명을 바꾸지 마세요.

### 3-4. 에러 코드

| 코드 | HTTP | 상황 |
| --- | --- | --- |
| `AI_001` | 500 | 서버 내부 오류 |
| `AI_002` | 400 | 요청 검증 실패 (`message`에 필드명 포함) |
| `AI_003` | 401 | 내부 키 불일치 |
| `AI_010` | 502 | 임베딩 생성 실패 |
| `AI_011` | 404 | 임베딩 미생성 (포지션 등록 시 호출 누락) |
| `AI_012` / `AI_013` / `AI_014` | 502/504/502 | LLM 실패 / 타임아웃 / 응답 형식 오류 |
| `AI_020` | 404 | 추천할 후보 없음 |
| `AI_030` | 502 | 스프링 호출 실패 |

스프링은 이 코드를 그대로 노출하지 말고 자기 도메인 코드로 감싸는 편이 낫습니다. (사용자에게 `AI_012`는 의미가 없습니다)

### 3-5. 엔드포인트

| 메서드 | 경로 | 언제 호출 |
| --- | --- | --- |
| PUT | `/api/v1/embeddings/freelancers` | 이력서·조건 저장 시 |
| PUT | `/api/v1/embeddings/positions` | 프로젝트 포지션 등록·수정 시 |
| GET | `/api/v1/embeddings/positions/{id}/candidates?limit=` | 1차 후보 풀 조회 |
| POST | `/api/v1/matchings/recommendations` | 매칭 라운드 시작 시 |

### 3-6. 스프링 쪽 호출 예시

```java
// global/infrastructure/ai/AiServerClient.java 같은 자리
private final RestClient restClient;   // baseUrl = ai.base-url

public MatchingResult recommend(Long positionId, int recruitCount) {
    return restClient.post()
            .uri("/api/v1/matchings/recommendations")
            .header("X-Internal-Api-Key", aiSettings.getInternalApiKey())
            .header("X-Trace-Id", MDC.get("traceId"))
            .body(new RecommendRequest(positionId, recruitCount))
            .retrieve()
            .body(new ParameterizedTypeReference<ApiResponse<MatchingResult>>() {})
            .data();
}
```

포트(`AiRecommendationPort`)로 감싸고 어댑터에 두면 기존 헥사고날 구조와 맞습니다.

---

## 4. DB 소유권 (제일 자주 사고 나는 곳)

같은 PostgreSQL을 두 서버가 봅니다. **테이블 단위로 쓰기 주인을 하나만 둡니다.**

| 테이블 | 쓰기 | 읽기 |
| --- | --- | --- |
| `freelancer_embedding`, `position_embedding` | AI 서버 | AI 서버 |
| `account`, `project_position`, `matching_*` 등 | 스프링 | AI 서버(읽기만) |

AI 서버가 스프링 소유 테이블의 상태를 바꿔야 하면 **스프링 API를 호출**합니다(`clients/spring.py`). 직접 UPDATE하면 도메인 규칙(상태 전이, 알림, 정산)을 우회하게 되고, 두 서버가 같은 행을 쓰는 순간 원인 못 찾는 버그가 생깁니다.

스키마는 SQL 파일이 단일 소스입니다. `Base.metadata.create_all()`을 쓰지 마세요 — 스프링의 `ddl-auto=validate`와 같은 이유로, 코드가 테이블을 만들기 시작하면 SQL 파일과 조용히 어긋납니다.

> 참고: 백엔드 스키마 주석에는 임베딩을 "FastAPI + ChromaDB로 이관"이라고 적혀 있습니다. pgvector로 방향이 바뀌었으니 그 주석도 정리하는 게 좋습니다.

---

## 5. Gemini 모델 규칙

키는 하나, 모델은 용도별로 설정에서만 정합니다.

```python
gemini.model_for(GeminiTask.EMBEDDING)   # settings.gemini_model_embedding
gemini.model_for(GeminiTask.MATCHING)    # settings.gemini_model_matching
gemini.model_for(GeminiTask.REVIEW)      # settings.gemini_model_review
```

- **모델명을 코드에 직접 쓰지 않습니다.** 흩어지면 어디를 바꿔야 모델이 바뀌는지 아무도 모르게 됩니다.
- 임베딩 모델을 바꾸면 차원이 달라집니다. `EMBEDDING_DIMENSION`, `vector(768)` 컬럼, **기존 데이터 재생성**이 세트입니다.
- 응답은 `response_schema`로 구조를 강제합니다. 자유 텍스트를 파싱하면 프롬프트 한 줄 고칠 때마다 깨집니다.
- LLM이 없는 ID를 지어낼 수 있으므로, 결과는 항상 입력 후보 풀로 걸러서 씁니다. (`MatchingService` 참고)

---

## 6. 새 도메인 추가 절차

1. `app/domains/{이름}/` 생성 — `router.py`, `service.py`, `schemas.py` (+필요 시 `repository.py`, `models.py`)
2. 에러가 필요하면 `core/errors.py`의 `AiErrorCode`에 추가 (번호대를 도메인별로 나눠 씀)
3. `app/api/v1/__init__.py`에 라우터 등록 — 내부 인증은 자동 적용됨
4. 테이블이 필요하면 `db/init/`에 SQL 추가 (AI 소유 테이블만)
5. `tests/`에 계약 테스트 추가
6. 스프링이 호출한다면 이 README의 3-5 표와 백엔드 레포 `.ai/API.md`를 함께 갱신

---

## 7. 아직 없는 것

스켈레톤 단계라 다음은 비어 있거나 stub입니다.

- `MatchingService._build_prompt` — 포지션 요구조건·프리랜서 요약을 실제로 채워야 합니다
- `ai_agent_log` 적재 (요청/응답/토큰 사용량 기록)
- 프로젝트 등록 AI 검수(`review` 모델 용도)
- 임베딩 배치 재생성, 고아 행 정리
- 인증 실패·LLM 실패에 대한 알림/메트릭
