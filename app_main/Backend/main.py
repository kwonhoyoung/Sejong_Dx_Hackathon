"""
Perplexity API만을 사용한 이슈 검색 시스템
구조화된 응답(response_format)을 활용한 통합 처리
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

# 환경 변수 로드
load_dotenv()


# ============= 설정 관리 =============
@dataclass
class Settings:
    """중앙 집중식 설정 관리 클래스."""

    # API Keys
    perplexity_api_key: Optional[str] = None

    # API Settings
    perplexity_base_url: str = "https://api.perplexity.ai/chat/completions"
    perplexity_model: str = "sonar"

    # Request Settings
    max_retries: int = 3
    timeout: int = 60
    retry_delay: int = 2
    max_concurrent_requests: int = 3
    request_delay: float = 0.5

    # Search Settings
    default_time_period: str = "최근 1주일"
    min_issues_required: int = 3
    max_issues_per_search: int = 7
    keywords_for_tags: int = 5

    # Response Settings
    max_tokens_search: int = 4000
    max_tokens_summary: int = 2000
    max_tokens_analysis: int = 2500

    @classmethod
    def from_env(cls) -> 'Settings':
        """환경 변수에서 설정을 로드합니다."""
        return cls(
            perplexity_api_key=os.getenv('PERPLEXITY_API_KEY'),
            perplexity_model=os.getenv('PERPLEXITY_MODEL', 'sonar'),
        )


@lru_cache()
def get_settings() -> Settings:
    """설정 싱글톤을 반환합니다."""
    return Settings.from_env()


# ============= 로깅 설정 =============
def setup_logging() -> None:
    """로깅 설정을 초기화합니다."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=''),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True
    )


setup_logging()

# 환경 변수 로드 확인
if os.getenv('DEBUG_MODE', '').lower() == 'true':
    settings = get_settings()
    logger.info(
        f"Perplexity API Key 설정: "
        f"{'Yes' if settings.perplexity_api_key else 'No'}"
    )
    if settings.perplexity_api_key:
        key_preview = (
            settings.perplexity_api_key[:10] + "..."
            if len(settings.perplexity_api_key) > 10
            else "Too short"
        )
        logger.info(f"Perplexity API Key Preview: {key_preview}")


# ============= 커스텀 예외 =============
class IssueSearchError(Exception):
    """이슈 검색 관련 기본 예외 클래스."""
    pass


class APIError(IssueSearchError):
    """API 호출 관련 예외 클래스."""
    pass


class ParsingError(IssueSearchError):
    """데이터 파싱 관련 예외 클래스."""
    pass


class ValidationError(IssueSearchError):
    """데이터 검증 관련 예외 클래스."""
    pass


# ============= 데이터 모델 =============
class IssueCategory(str, Enum):
    """이슈 카테고리 열거형."""

    NEWS = "뉴스"
    TECH = "기술"
    BUSINESS = "비즈니스"
    RESEARCH = "연구"
    POLICY = "정책"
    GENERAL = "일반"


@dataclass
class Issue:
    """이슈 정보를 담는 데이터 클래스."""

    title: str
    url: str
    source: str = "Unknown"
    published_date: Optional[str] = None
    category: str = IssueCategory.GENERAL
    description: Optional[str] = None
    summary: Optional[str] = None

    def is_valid_summary(self) -> bool:
        """유효한 요약인지 확인합니다."""
        return bool(self.summary and len(self.summary) >= 50)


@dataclass
class SearchResult:
    """검색 결과를 담는 데이터 클래스."""

    topic: str
    keywords: List[str]
    issues: List[Issue]
    search_time: float
    total_found: int = field(init=False)

    def __post_init__(self):
        """초기화 후 처리를 수행합니다."""
        self.total_found = len(self.issues)


@dataclass
class AnalysisResult:
    """분석 결과를 담는 데이터 클래스."""

    summary: str
    insights: List[str] = field(default_factory=list)
    future_outlook: Optional[str] = None
    analyzed_count: int = 0

    def to_full_text(self) -> str:
        """전체 분석 내용을 텍스트로 변환합니다."""
        parts = [self.summary, "\n주요 인사이트:"]
        parts.extend(f"- {insight}" for insight in self.insights)
        if self.future_outlook:
            parts.append(f"\n향후 전망: {self.future_outlook}")
        return "\n".join(parts)


# ============= HTTP 클라이언트 관리 =============
class HTTPClientManager:
    """HTTP 클라이언트를 관리하는 싱글톤 클래스."""

    _client: Optional[httpx.AsyncClient] = None

    @classmethod
    async def get_client(cls) -> httpx.AsyncClient:
        """HTTP 클라이언트 인스턴스를 반환합니다."""
        if cls._client is None:
            timeout_config = httpx.Timeout(
                get_settings().timeout,
                read=60.0,
                connect=10.0
            )
            cls._client = httpx.AsyncClient(
                timeout=timeout_config,
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10
                )
            )
        return cls._client

    @classmethod
    async def close(cls) -> None:
        """HTTP 클라이언트를 닫습니다."""
        if cls._client:
            await cls._client.aclose()
            cls._client = None


# ============= Perplexity 클라이언트 =============
class PerplexityClient:
    """Perplexity API와 통신하는 클라이언트 클래스."""

    def __init__(self, api_key: Optional[str] = None):
        """Perplexity 클라이언트를 초기화합니다."""
        settings = get_settings()
        self.api_key = api_key or settings.perplexity_api_key
        self.base_url = settings.perplexity_base_url
        self.model = settings.perplexity_model

    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.3,
        response_format: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perplexity API에 요청을 보냅니다."""
        if not self.api_key:
            raise APIError("Perplexity API 키가 설정되지 않았습니다.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        if response_format:
            payload["response_format"] = response_format

        settings = get_settings()

        for attempt in range(settings.max_retries):
            try:
                if attempt > 0:
                    delay = settings.retry_delay * (attempt + 1)
                    logger.info(f"재시도 전 {delay}초 대기 중...")
                    await asyncio.sleep(delay)

                client = await HTTPClientManager.get_client()
                response = await client.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    logger.error(
                        "Perplexity API 인증 실패. API 키를 확인해주세요."
                    )
                elif e.response.status_code == 429:
                    logger.warning(
                        "API 속도 제한에 도달했습니다. 잠시 후 재시도합니다."
                    )
                else:
                    logger.error(
                        f"Perplexity API HTTP 에러: {e.response.status_code}"
                    )

                if attempt < settings.max_retries - 1:
                    continue
                else:
                    raise APIError(f"API 요청 실패: {e.response.status_code}")

            except Exception as e:
                logger.error(f"Perplexity API 요청 중 알 수 없는 오류: {e}")
                if attempt < settings.max_retries - 1:
                    continue
                else:
                    raise APIError(f"API 요청 중 오류 발생: {e}")

    async def search_issues_with_summaries(
        self,
        topic: str,
        time_period: str
    ) -> List[Issue]:
        """이슈를 검색하고 각 이슈의 요약을 포함하여 반환합니다."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "issues": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "url": {"type": "string"},
                                    "source": {"type": "string"},
                                    "published_date": {"type": "string"},
                                    "category": {"type": "string"},
                                    "description": {"type": "string"},
                                    "summary": {"type": "string"}
                                },
                                "required": [
                                    "title", "url", "source",
                                    "published_date", "category",
                                    "description", "summary"
                                ]
                            }
                        }
                    },
                    "required": ["issues"]
                }
            }
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert news researcher who searches for "
                    "articles and provides comprehensive summaries in Korean."
                )
            },
            {
                "role": "user",
                "content": f"""
주제: "{topic}"
기간: {time_period}

위 주제에 대한 최신 뉴스 기사를 {get_settings().max_issues_per_search}개 찾아주세요.

각 기사에 대해 다음 정보를 포함해주세요:
1. title: 정확한 기사 제목
2. url: 전체 URL (https://로 시작)
3. source: 언론사 이름
4. published_date: YYYY-MM-DD 형식
5. category: news/technology/business/research/policy 중 하나
6. description: 기사의 주요 내용 50-150단어 요약 (한국어)
7. summary: 기사의 상세한 요약 200-300단어 (한국어) - 핵심 내용, 중요한 통계나 수치, 시사점 포함

신뢰할 수 있는 주요 언론사의 기사를 우선적으로 선택하고, 다양한 출처에서 가져오세요.
"""
            }
        ]

        try:
            response = await self._make_request(
                messages=messages,
                max_tokens=get_settings().max_tokens_search,
                temperature=0.2,
                response_format=response_format
            )

            content = response['choices'][0]['message']['content']
            data = json.loads(content)

            issues = []
            for item in data.get('issues', []):
                issues.append(Issue(**item))

            return issues

        except Exception as e:
            logger.error(f"이슈 검색 및 요약 실패: {e}")
            return []

    async def generate_keywords(
        self,
        topic: str,
        max_keywords: int
    ) -> List[str]:
        """주제에 대한 키워드를 생성합니다."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["keywords"]
                }
            }
        }

        messages = [
            {
                "role": "system",
                "content": "You are an expert in generating relevant search keywords."
            },
            {
                "role": "user",
                "content": (
                    f'주제 "{topic}"에 대한 핵심 검색 키워드를 '
                    f'{max_keywords}개 생성해주세요. '
                    f'다양하고 관련성 높은 키워드를 선택하세요.'
                )
            }
        ]

        try:
            response = await self._make_request(
                messages=messages,
                max_tokens=200,
                temperature=0.7,
                response_format=response_format
            )

            content = response['choices'][0]['message']['content']
            data = json.loads(content)
            keywords = data.get('keywords', [])

            # 중복 제거 및 원래 주제 포함
            unique_keywords = list(dict.fromkeys([topic] + keywords))
            return unique_keywords[:max_keywords]

        except Exception as e:
            logger.error(f"키워드 생성 실패: {e}")
            return [topic]

    async def analyze_issues(
        self,
        issues: List[Issue],
        topic: str
    ) -> Optional[AnalysisResult]:
        """여러 이슈를 종합적으로 분석합니다."""
        if not issues:
            return None

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "trend_summary": {"type": "string"},
                        "insights": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "future_outlook": {"type": "string"}
                    },
                    "required": ["trend_summary", "insights", "future_outlook"]
                }
            }
        }

        # 이슈 정보 준비
        issues_text = "\n\n".join(
            f"[{i+1}] {issue.title}\n"
            f"출처: {issue.source}\n"
            f"요약: {issue.summary[:200]}..."
            for i, issue in enumerate(issues[:10])
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a technology trend analyst who provides "
                    "comprehensive insights in Korean."
                )
            },
            {
                "role": "user",
                "content": f"""
주제: "{topic}"

다음은 관련 이슈들입니다:
{issues_text}

위 이슈들을 종합적으로 분석하여 다음을 제공해주세요:
1. trend_summary: 전체 트렌드를 관통하는 핵심 내용 2-3문장 요약 (한국어)
2. insights: 주요 인사이트 3-5개 (각각 한국어로 1-2문장)
3. future_outlook: 이러한 트렌드를 바탕으로 한 향후 전망 2-3문장 (한국어)

심도 있는 분석과 실용적인 인사이트를 제공해주세요.
"""
            }
        ]

        try:
            response = await self._make_request(
                messages=messages,
                max_tokens=get_settings().max_tokens_analysis,
                temperature=0.5,
                response_format=response_format
            )

            content = response['choices'][0]['message']['content']
            data = json.loads(content)

            return AnalysisResult(
                summary=data.get('trend_summary', '분석 결과 없음'),
                insights=data.get('insights', []),
                future_outlook=data.get('future_outlook'),
                analyzed_count=len(issues)
            )

        except Exception as e:
            logger.error(f"이슈 분석 실패: {e}")
            return None


# ============= 이슈 검색 서비스 =============
class IssueSearchService:
    """이슈 검색 및 분석을 수행하는 서비스 클래스."""

    def __init__(self):
        """검색 서비스를 초기화합니다."""
        self.perplexity = PerplexityClient()
        self.settings = get_settings()
        self.semaphore = asyncio.Semaphore(
            self.settings.max_concurrent_requests
        )

    async def search(
        self,
        topic: str,
        time_period: Optional[str] = None,
        analyze: bool = True
    ) -> Dict[str, Any]:
        """통합 검색 및 분석을 수행합니다."""
        start_time = time.time()
        time_period = time_period or self.settings.default_time_period

        try:
            # 1. 이슈 검색 및 요약
            logger.info(f"'{topic}' 주제로 이슈 검색 및 요약 시작")
            issues = await self.perplexity.search_issues_with_summaries(
                topic, time_period
            )

            if not issues:
                logger.warning("이슈를 찾지 못했습니다")
                return self._create_empty_result(
                    topic, time.time() - start_time
                )

            logger.info(f"{len(issues)}개 이슈 검색 완료")

            # 2. 키워드 생성
            keywords = await self.perplexity.generate_keywords(
                topic, self.settings.keywords_for_tags
            )

            # 3. 종합 분석 (선택적)
            analysis = None
            if analyze and len(issues) >= 2:
                logger.info("종합 분석 수행 중...")
                analysis = await self.perplexity.analyze_issues(issues, topic)

            search_time = time.time() - start_time
            logger.info(
                f"전체 프로세스 완료: {len(issues)}개 이슈, "
                f"{search_time:.2f}초 소요"
            )

            return {
                "search_result": SearchResult(
                    topic=topic,
                    keywords=keywords,
                    issues=issues,
                    search_time=search_time
                ),
                "analysis": analysis
            }

        except Exception as e:
            logger.exception(f"검색 프로세스 중 오류 발생: {e}")
            return self._create_error_result(
                topic, time.time() - start_time, str(e)
            )

    def _create_empty_result(
        self,
        topic: str,
        search_time: float
    ) -> Dict[str, Any]:
        """빈 결과를 생성합니다."""
        return {
            "search_result": SearchResult(
                topic=topic,
                keywords=[topic],
                issues=[],
                search_time=search_time
            ),
            "analysis": None
        }

    def _create_error_result(
        self,
        topic: str,
        search_time: float,
        error: str
    ) -> Dict[str, Any]:
        """에러 결과를 생성합니다."""
        result = self._create_empty_result(topic, search_time)
        result["error"] = error
        return result


# ============= API 스키마 및 FastAPI 앱 =============
class SearchRequest(BaseModel):
    """검색 요청 스키마."""

    topic: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="검색할 주제"
    )
    time_period: str = Field(
        default="최근 1주일",
        description="검색 기간"
    )
    analyze: bool = Field(
        default=True,
        description="분석 수행 여부"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기를 관리합니다."""
    logger.info("애플리케이션 시작")

    # API 키 상태 확인
    settings = get_settings()
    logger.info(
        f"Perplexity API Key: "
        f"{'Configured' if settings.perplexity_api_key else 'Not configured'}"
    )

    app.state.search_service = IssueSearchService()
    yield
    logger.info("애플리케이션 종료")
    await HTTPClientManager.close()


app = FastAPI(
    title="Perplexity AI Issue Search API",
    version="1.0.0",
    description=(
        "Perplexity API의 구조화된 응답을 활용한 "
        "통합 이슈 검색 및 분석 시스템"
    ),
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """루트 엔드포인트입니다."""
    return {"message": "Perplexity AI Issue Search API v1.0.0"}


@app.post("/api/search")
async def search_issues_endpoint(
    request: Request,
    search_request: SearchRequest
):
    """이슈 검색 엔드포인트입니다."""
    try:
        search_service: IssueSearchService = request.app.state.search_service
        result = await search_service.search(
            topic=search_request.topic,
            time_period=search_request.time_period,
            analyze=search_request.analyze
        )
        formatted_report = format_final_report(result, search_request.topic)
        return JSONResponse(content=formatted_report)
    except Exception as e:
        logger.exception(f"API 엔드포인트에서 처리되지 않은 예외 발생: {e}")
        raise HTTPException(
            status_code=500,
            detail="서버 내부 오류가 발생했습니다."
        )


def format_final_report(
    result: Dict[str, Any],
    topic: str
) -> Dict[str, Any]:
    """최종 보고서를 포맷팅합니다."""
    search_result = result.get("search_result")
    analysis = result.get("analysis")

    if not search_result or not search_result.issues:
        return {
            "제목": f"'{topic}'에 대한 검색 결과가 없습니다",
            "태그": [topic],
            "보고서": {
                "정리된 내용": "관련된 최신 이슈를 찾을 수 없었습니다.",
                "AI가 제공하는 리포트": "분석할 내용이 없습니다.",
                "출처 링크": []
            }
        }

    issues = search_result.issues
    keywords = search_result.keywords

    # 요약 내용 정리
    summarized_contents = []
    source_links = []
    seen_urls = set()

    for issue in issues:
        if issue.summary and issue.url not in seen_urls:
            summary_text = issue.summary.strip()
            summarized_contents.append(
                f"### {issue.title}\n\n{summary_text}"
            )

        if issue.url and issue.url not in seen_urls:
            source_links.append({
                "title": issue.title,
                "url": issue.url,
                "source": issue.source,
                "date": issue.published_date
            })
            seen_urls.add(issue.url)

    # 제목 설정
    if analysis and analysis.summary:
        title = analysis.summary
    else:
        title = f"'{topic}'에 대한 최신 동향 분석"

    # 태그 정리
    tags = [kw for kw in keywords if kw.lower() != topic.lower()]
    tags.insert(0, topic)
    tags = list(dict.fromkeys(tags))[:get_settings().keywords_for_tags]

    # 최종 보고서 구성
    report_content = {
        "정리된 내용": (
            "\n\n---\n\n".join(summarized_contents)
            if summarized_contents
            else "유효한 요약 내용이 없습니다."
        ),
        "AI가 제공하는 리포트": (
            analysis.to_full_text()
            if analysis
            else "종합 분석을 생성하지 못했습니다."
        ),
        "출처 링크": source_links
    }

    # 디버그 모드에서 통계 추가
    if os.getenv('DEBUG_MODE', '').lower() == 'true':
        report_content["통계"] = {
            "검색된 이슈": len(issues),
            "유효한 요약": len(summarized_contents),
            "검색 시간": f"{search_result.search_time:.2f}초"
        }

    return {
        "제목": title,
        "태그": tags,
        "보고서": report_content
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트입니다."""
    settings = get_settings()
    return {
        "status": "healthy",
        "api_key_configured": bool(settings.perplexity_api_key),
        "model": settings.perplexity_model
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)