"""
Improved Issue Search System with a Hybrid Approach (Perplexity + Claude)
개선된 하이브리드 이슈 검색 시스템 (오류 수정 버전)
"""

import asyncio
import re
import time
import json
from typing import List, Dict, Optional, Any, Protocol
from dataclasses import dataclass, field, fields # 'fields' 추가
from datetime import datetime
from contextlib import asynccontextmanager
from functools import lru_cache
from enum import Enum
import httpx
from loguru import logger
import anthropic
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 환경 변수 로드
load_dotenv()

# ============= 설정 관리 =============
@dataclass
class Settings:
    """중앙 집중식 설정 관리"""
    # API Keys
    perplexity_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # API Settings
    perplexity_base_url: str = "https://api.perplexity.ai/chat/completions"
    perplexity_model: str = "sonar-pro"
    claude_model: str = "claude-sonnet-4-20250514"

    # Request Settings
    max_retries: int = 3
    timeout: int = 60
    retry_delay: int = 2
    max_concurrent_requests: int = 2  # 3에서 2로 감소
    request_delay: float = 0.5  # 요청 간 지연 시간 추가

    # Search Settings
    default_time_period: str = "최근 1주일"
    min_issues_required: int = 3
    max_issues_per_search: int = 7
    keywords_for_tags: int = 5
    use_content_extraction: bool = True  # 콘텐츠 추출 사용 여부

    # Response Settings
    min_content_length: int = 100
    min_summary_length: int = 50
    max_tokens_search: int = 4000
    max_tokens_summary: int = 1500
    max_tokens_analysis: int = 1500

    @classmethod
    def from_env(cls) -> 'Settings':
        """환경 변수에서 설정 로드"""
        return cls(
            perplexity_api_key=os.getenv('PERPLEXITY_API_KEY'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            claude_model=os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-20250514'),
            use_content_extraction=os.getenv('USE_CONTENT_EXTRACTION', 'true').lower() == 'true',
        )

@lru_cache()
def get_settings() -> Settings:
    """설정 싱글톤"""
    return Settings.from_env()

# ============= 로깅 설정 =============
def setup_logging():
    """로깅 설정 초기화"""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=''),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )

setup_logging()

# 환경 변수 로드 확인 (디버깅용)
if os.getenv('DEBUG_MODE', '').lower() == 'true':
    settings = get_settings()
    logger.info(f"Perplexity API Key 설정: {'Yes' if settings.perplexity_api_key else 'No'}")
    logger.info(f"Anthropic API Key 설정: {'Yes' if settings.anthropic_api_key else 'No'}")
    logger.info(f"콘텐츠 추출 사용: {'Yes' if settings.use_content_extraction else 'No'}")
    if settings.perplexity_api_key:
        # API 키의 일부만 표시 (보안)
        key_preview = settings.perplexity_api_key[:10] + "..." if len(settings.perplexity_api_key) > 10 else "Too short"
        logger.info(f"Perplexity API Key Preview: {key_preview}")

# ============= 커스텀 예외 =============
class IssueSearchError(Exception): pass
class APIError(IssueSearchError): pass
class ParsingError(IssueSearchError): pass
class ValidationError(IssueSearchError): pass

# ============= 데이터 모델 =============
class IssueCategory(str, Enum):
    NEWS = "뉴스"
    TECH = "기술"
    BUSINESS = "비즈니스"
    RESEARCH = "연구"
    POLICY = "정책"
    GENERAL = "일반"

@dataclass
class Issue:
    title: str
    url: str
    source: str = "Unknown"
    published_date: Optional[str] = None
    category: str = IssueCategory.GENERAL
    description: Optional[str] = None  # 검색 시 얻은 간단한 설명
    content: Optional[str] = None
    summary: Optional[str] = None

    def is_valid_summary(self) -> bool:
        return bool(
            self.summary and
            len(self.summary) >= get_settings().min_summary_length and
            not any(self.summary.startswith(prefix) for prefix in ["실패", "콘텐츠", "처리 중", "요약할 수"])
        )

@dataclass
class SearchResult:
    topic: str
    keywords: List[str]
    issues: List[Issue]
    search_time: float
    total_found: int = field(init=False)

    def __post_init__(self):
        self.total_found = len(self.issues)

@dataclass
class AnalysisResult:
    summary: str
    insights: List[str] = field(default_factory=list)
    future_outlook: Optional[str] = None
    analyzed_count: int = 0

    def to_full_text(self) -> str:
        parts = [self.summary, "\n주요 인사이트:"]
        parts.extend(f"- {insight}" for insight in self.insights)
        if self.future_outlook:
            parts.append(f"\n향후 전망: {self.future_outlook}")
        return "\n".join(parts)

# ============= 프로토콜(인터페이스) 정의 =============
class KeywordGenerator(Protocol):
    async def generate_keywords(self, topic: str, max_keywords: int) -> List[str]: ...

class ContentExtractor(Protocol):
    async def extract_content(self, url: str) -> Optional[str]: ...

# ============= HTTP 클라이언트 관리 =============
class HTTPClientManager:
    _client: Optional[httpx.AsyncClient] = None

    @classmethod
    async def get_client(cls) -> httpx.AsyncClient:
        if cls._client is None:
            # ### 수정점: 명시적인 read/connect 타임아웃 설정 ###
            timeout_config = httpx.Timeout(
                get_settings().timeout,  # 전체 타임아웃
                read=60.0,               # 읽기 타임아웃 (서버 응답 대기 시간)
                connect=10.0             # 연결 타임아웃
            )
            cls._client = httpx.AsyncClient(
                timeout=timeout_config,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
            )
        return cls._client

    @classmethod
    async def close(cls):
        if cls._client:
            await cls._client.aclose()
            cls._client = None

# ============= Perplexity 클라이언트 (하이브리드 접근법 적용) =============
class PerplexityClient:
    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.perplexity_api_key
        self.base_url = settings.perplexity_base_url
        self.model = settings.perplexity_model

    async def _make_request(self, prompt: str, max_tokens: int, temperature: float, system_message: str) -> Dict[str, Any]:
        if not self.api_key:
            raise APIError("Perplexity API 키가 설정되지 않았습니다.")

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature
        }

        settings = get_settings()

        for attempt in range(settings.max_retries):
            try:
                # 요청 간 지연 (첫 번째 시도 제외)
                if attempt > 0:
                    delay = settings.retry_delay * (attempt + 1)
                    logger.info(f"재시도 전 {delay}초 대기 중...")
                    await asyncio.sleep(delay)

                client = await HTTPClientManager.get_client()
                response = await client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    logger.error("Perplexity API 인증 실패. API 키를 확인해주세요.")
                    if attempt == 0:  # 첫 번째 시도에서만 상세 정보 출력
                        logger.error(f"API Key Preview: {self.api_key[:10]}..." if self.api_key and len(self.api_key) > 10 else "Invalid key")
                elif e.response.status_code == 429:
                    logger.warning("API 속도 제한에 도달했습니다. 잠시 후 재시도합니다.")
                else:
                    logger.error(f"Perplexity API HTTP 에러: {e.response.status_code}")

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

    async def search_issues_structured(self, topic: str, time_period: str) -> List[Issue]:
        prompt = self._build_json_search_prompt(topic, time_period)
        system_message = "You are a helpful assistant that finds relevant news articles and returns the information in a structured JSON format."

        try:
            response = await self._make_request(
                prompt, max_tokens=get_settings().max_tokens_search, temperature=0.2, system_message=system_message
            )
            content = response['choices'][0]['message']['content']
            return self._parse_issues_from_json(content)
        except Exception as e:
            logger.error(f"구조화된 이슈 검색 실패: {e}")
            return []

    def _build_json_search_prompt(self, topic: str, time_period: str) -> str:
        max_issues = get_settings().max_issues_per_search
        return f"""
        Find {max_issues} recent and relevant news articles or reports about the topic: "{topic}".
        Search within the time period: "{time_period}".
        For each article, provide the title, a valid full URL, the source name, the publication date, and a brief description or excerpt.
        
        Respond ONLY with a single JSON array of objects. Do not include any other text or explanation.
        The JSON format should be:
        [
          {{
            "title": "Article Title",
            "url": "https://example.com/article-1",
            "source": "Example News",
            "published_date": "YYYY-MM-DD",
            "category": "뉴스/기술/비즈니스/연구/정책 중 하나",
            "description": "Brief description or excerpt from the article (50-150 words)"
          }}
        ]
        """

    # ### 수정점: 데이터 안정성 강화를 위한 파싱 로직 개선 ###
    def _parse_issues_from_json(self, content: str) -> List[Issue]:
        """AI가 생성한 JSON 문자열을 파싱. url 타입 검증 및 예기치 않은 필드 처리 추가."""
        try:
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if not match:
                logger.warning("응답에서 JSON 배열을 찾을 수 없습니다.")
                return []

            json_str = match.group(0)
            data = json.loads(json_str)

            issues = []
            issue_fields = {f.name for f in fields(Issue)}

            for item in data:
                # 1. url 필드가 존재하고, 그 값이 문자열인지 명시적으로 확인
                if isinstance(item, dict) and isinstance(item.get('url'), str):
                    # 2. API 응답에 예상치 못한 필드가 있어도 에러가 나지 않도록 필터링
                    filtered_item = {k: v for k, v in item.items() if k in issue_fields}
                    try:
                        issues.append(Issue(**filtered_item))
                    except TypeError as e:
                        logger.warning(f"Issue 객체 생성 실패: {e}. 데이터: {filtered_item}")
                else:
                    logger.warning(f"유효하지 않은 이슈 데이터 건너뜀: {item}")
            return issues
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}\n내용: {content}...")
            raise ParsingError("AI 응답의 JSON 형식이 잘못되었습니다.")
        except Exception as e:
            logger.error(f"이슈 파싱 중 알 수 없는 오류: {e}")
            return []

    async def generate_keywords(self, topic: str, max_keywords: int) -> List[str]:
        prompt = f'주제: "{topic}"\n\n이 주제를 대표하는 핵심 검색 키워드를 {max_keywords}개 생성해주세요. 한 줄에 하나씩 키워드만 나열해주세요.'
        system_message = "You are an expert in generating search keywords."
        try:
            response = await self._make_request(prompt, max_tokens=200, temperature=0.7, system_message=system_message)
            content = response['choices'][0]['message']['content']
            keywords = [re.sub(r'^\s*[-\d.]\s*', '', kw).strip() for kw in content.strip().split('\n') if kw.strip()]
            return list(dict.fromkeys(keywords))
        except Exception as e:
            logger.error(f"키워드 생성 실패: {e}")
            return [topic]

    async def extract_content(self, url: str) -> Optional[str]:
        """웹페이지에서 콘텐츠 추출 (개선된 프롬프트)"""
        prompt = f"""웹 URL: {url}

위 URL의 웹페이지를 방문하여 기사의 전체 본문 내용을 추출해주세요. 
기사의 제목, 부제목, 본문 내용을 모두 포함하되, 광고, 네비게이션 메뉴, 관련 기사 링크 등은 제외해주세요.

만약 웹페이지에 접근할 수 없거나 콘텐츠를 찾을 수 없다면 "CONTENT_NOT_FOUND"라고 응답해주세요.
그렇지 않다면 추출한 기사 본문을 그대로 출력해주세요."""

        system_message = "You are a web content extraction tool. Extract the main article content from the given URL."

        try:
            response = await self._make_request(prompt, max_tokens=3000, temperature=0.1, system_message=system_message)
            content = response['choices'][0]['message']['content'].strip()

            # 콘텐츠 추출 실패 확인
            if content == "CONTENT_NOT_FOUND" or len(content) < 50:
                logger.warning(f"콘텐츠 추출 실패 또는 너무 짧음 ({url}): {len(content)} 글자")
                return None

            # 콘텐츠 검증
            if "제공해주신 내용" in content or "요약할 수 없습니다" in content:
                logger.warning(f"잘못된 콘텐츠 추출 ({url})")
                return None

            return content if len(content) >= get_settings().min_content_length else None

        except Exception as e:
            logger.error(f"콘텐츠 추출 실패 ({url}): {e}")
            return None

# ============= Claude 클라이언트 (변경 없음) =============
class ClaudeClient:
    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.anthropic_api_key
        self.model = settings.claude_model
        self.client = None
        if self.api_key:
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

    async def summarize_content(self, content: str, topic: str) -> str:
        if not self.client:
            return "Claude API 키가 없어 요약할 수 없습니다."

        clean_content = re.sub(r'\s+', ' ', content).strip()

        # 콘텐츠 길이 검증 강화
        if len(clean_content) < get_settings().min_content_length:
            logger.warning(f"콘텐츠가 너무 짧음: {len(clean_content)} 글자")
            return "콘텐츠가 충분하지 않아 요약할 수 없습니다."

        # 콘텐츠 품질 검증
        if any(phrase in clean_content for phrase in ["CONTENT_NOT_FOUND", "페이지를 찾을 수 없", "404", "Not Found"]):
            logger.warning("콘텐츠에 에러 메시지가 포함됨")
            return "유효한 기사 콘텐츠를 찾을 수 없습니다."

        prompt = f"""주제: "{topic}"

다음은 추출된 기사 본문입니다. 이 내용을 바탕으로 핵심 내용을 한국어로 3-5 문단으로 요약해주세요.
만약 제공된 텍스트가 실제 기사 내용이 아니거나 요약하기에 불충분하다면, "기사 내용이 불충분하여 요약할 수 없습니다"라고 응답하세요.

--- 기사 본문 시작 ---
{clean_content}
--- 기사 본문 끝 ---

위 내용을 바탕으로 객관적이고 정확한 요약을 작성해주세요."""

        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=get_settings().max_tokens_summary,
                temperature=0.3,
                system="You are an expert Korean news summarizer. Create concise, accurate summaries based on the provided content only. If the content is insufficient or not a valid article, say so clearly.",
                messages=[{"role": "user", "content": prompt}]
            )
            summary = message.content[0].text.strip()

            # 요약 품질 검증
            if any(phrase in summary for phrase in ["제공해주신 내용", "요약할 수 없습니다", "기사 내용이 불충분"]):
                logger.warning(f"요약 품질 문제: {summary}...")
                return "기사 콘텐츠 추출에 실패했습니다."

            return summary if len(summary) > get_settings().min_summary_length else "요약 생성에 실패했습니다."

        except Exception as e:
            logger.error(f"Claude 요약 실패: {e}")
            return "요약 생성 중 오류가 발생했습니다."

    async def analyze_issues(self, issues: List[Issue], topic: str) -> Optional[AnalysisResult]:
        if not self.client: return None
        valid_issues = [issue for issue in issues if issue.is_valid_summary()]
        if not valid_issues: return None

        issues_text = "\n\n".join(f"[{i+1}] {issue.title}\n{issue.summary}..." for i, issue in enumerate(valid_issues[:10]))
        prompt = f"""주제: "{topic}"

다음은 관련 이슈들의 요약본입니다:
{issues_text}

위 이슈들을 종합적으로 분석하여 아래 JSON 형식에 맞춰 한국어로 응답해주세요:
{{
    "trend_summary": "전체 트렌드를 관통하는 핵심적인 내용 1~2 문장 요약",
    "insights": [
        "주요 인사이트 1 (핵심적인 발견 또는 시사점)",
        "주요 인사이트 2 (핵심적인 발견 또는 시사점)",
        "주요 인사이트 3 (핵심적인 발견 또는 시사점)"
    ],
    "future_outlook": "이러한 트렌드를 바탕으로 한 향후 전망 1~2 문장"
}}"""

        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=get_settings().max_tokens_analysis,
                temperature=0.5,
                system="You are a tech trend analysis expert who provides insights in JSON format.",
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = message.content[0].text.strip()

            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()

            data = json.loads(response_text)
            return AnalysisResult(
                summary=data.get('trend_summary', '분석 결과 없음'),
                insights=data.get('insights', []),
                future_outlook=data.get('future_outlook'),
                analyzed_count=len(valid_issues)
            )
        except Exception as e:
            logger.error(f"Claude 분석 실패: {e}")
            return None

# ============= 이슈 검색 서비스 (하이브리드 로직 적용) =============
class IssueSearchService:
    def __init__(self):
        self.perplexity = PerplexityClient()
        self.claude = ClaudeClient()
        self.settings = get_settings()
        self.semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)

    async def search(self, topic: str, time_period: Optional[str] = None, analyze: bool = True) -> Dict[str, Any]:
        start_time = time.time()
        time_period = time_period or self.settings.default_time_period

        try:
            logger.info(f"'{topic}' 주제로 구조화된 이슈 검색 시작")
            issues = await self._collect_issues_with_retry(topic, time_period)

            if not issues:
                logger.warning("이슈를 찾지 못했습니다")
                return self._create_empty_result(topic, time.time() - start_time)

            logger.info(f"{len(issues)}개 이슈 콘텐츠 보강 및 요약 중...")
            await self._enrich_issues_content(issues, topic)

            analysis = None
            if analyze and any(issue.is_valid_summary() for issue in issues):
                logger.info("종합 분석 수행 중...")
                analysis = await self.claude.analyze_issues(issues, topic)

            keywords = await self.perplexity.generate_keywords(topic, self.settings.keywords_for_tags)

            search_time = time.time() - start_time
            logger.info(f"검색 완료: {len(issues)}개 이슈, {search_time:.2f}초 소요")

            return {
                "search_result": SearchResult(topic=topic, keywords=keywords, issues=issues, search_time=search_time),
                "analysis": analysis
            }
        except Exception as e:
            logger.exception(f"검색 프로세스 중 심각한 오류 발생: {e}")
            return self._create_error_result(topic, time.time() - start_time, str(e))

    async def _collect_issues_with_retry(self, topic: str, time_period: str) -> List[Issue]:
        all_issues: List[Issue] = []
        for attempt in range(self.settings.max_retries):
            try:
                logger.info(f"이슈 검색 시도 ({attempt + 1}/{self.settings.max_retries})")
                new_issues = await self.perplexity.search_issues_structured(topic, time_period)

                if new_issues:
                    existing_urls = {issue.url for issue in all_issues}
                    unique_new_issues = [issue for issue in new_issues if issue.url not in existing_urls]

                    if unique_new_issues:
                        all_issues.extend(unique_new_issues)
                        logger.info(f"새로운 이슈 {len(unique_new_issues)}개 추가 (총 {len(all_issues)}개)")
                    else:
                        logger.info("모든 이슈가 중복됨. 검색 종료.")
                        break

                if len(all_issues) >= self.settings.min_issues_required:
                    logger.info(f"충분한 이슈 수집 완료: {len(all_issues)}개")
                    break

                if attempt < self.settings.max_retries - 1:
                    await asyncio.sleep(self.settings.retry_delay)

            except (APIError, ParsingError) as e:
                logger.warning(f"이슈 수집 실패 (시도 {attempt + 1}): {e}")
                if attempt < self.settings.max_retries - 1:
                    await asyncio.sleep(self.settings.retry_delay * (attempt + 1))
                else:
                    logger.error("최대 재시도 횟수 초과. 이슈 수집을 중단합니다.")
                    break

        return all_issues[:get_settings().max_issues_per_search]

    async def _enrich_issues_content(self, issues: List[Issue], topic: str):
        tasks = []
        for i, issue in enumerate(issues):
            if issue.url:
                # 요청 간 지연을 위해 각 태스크에 지연 시간 추가
                delay = i * self.settings.request_delay
                tasks.append(self._enrich_single_issue_with_delay(issue, topic, delay))

        if tasks:
            await asyncio.gather(*tasks)

    async def _enrich_single_issue_with_delay(self, issue: Issue, topic: str, delay: float):
        """지연 시간을 포함한 단일 이슈 보강"""
        if delay > 0:
            await asyncio.sleep(delay)
        await self._enrich_single_issue(issue, topic)

    async def _enrich_single_issue(self, issue: Issue, topic: str):
        async with self.semaphore:
            try:
                # 콘텐츠 추출 옵션 확인
                if self.settings.use_content_extraction:
                    # 콘텐츠 추출 시도
                    content = await self.perplexity.extract_content(issue.url)

                    if content and len(content) >= self.settings.min_content_length:
                        issue.content = content
                        # Claude로 요약 생성
                        summary = await self.claude.summarize_content(issue.content, topic)
                        issue.summary = summary

                        # 요약 실패 확인
                        if "추출에 실패" in summary or "요약할 수 없습니다" in summary:
                            logger.info(f"대체 요약 생성 시도: {issue.title}")
                            # description이 있으면 활용
                            if issue.description and len(issue.description) > 50:
                                issue.summary = f"{issue.title}\n\n{issue.description}\n\n(원문 콘텐츠 추출 실패로 검색 결과의 요약을 제공합니다)"
                            else:
                                issue.summary = f"'{issue.title}' 기사가 {issue.source}에 게재되었습니다. 상세 내용은 원문 링크를 참조하세요."
                        return  # 성공적으로 처리됨

                # 콘텐츠 추출을 사용하지 않거나 실패한 경우
                logger.info(f"Description 기반 요약 사용: {issue.title}")
                if issue.description and len(issue.description) > 50:
                    # description이 충분히 길면 Claude로 요약 생성
                    enhanced_summary = await self.claude.summarize_content(
                        f"제목: {issue.title}\n출처: {issue.source}\n내용: {issue.description}",
                        topic
                    )
                    # 요약이 성공적이면 사용, 아니면 description 그대로 사용
                    if "추출에 실패" not in enhanced_summary and "요약할 수 없습니다" not in enhanced_summary:
                        issue.summary = enhanced_summary
                    else:
                        issue.summary = f"{issue.title}\n\n{issue.description}"
                else:
                    issue.summary = f"'{issue.title}' 관련 기사입니다. ({issue.source})"

            except Exception as e:
                logger.error(f"이슈 보강 오류 ({issue.url}): {e}")
                # 에러 발생 시에도 description 활용
                if issue.description:
                    issue.summary = f"{issue.title}\n\n{issue.description}\n\n(처리 중 오류 발생)"
                else:
                    issue.summary = f"'{issue.title}' - 콘텐츠 처리 중 오류가 발생했습니다."

    def _create_empty_result(self, topic: str, search_time: float) -> Dict[str, Any]:
        return {
            "search_result": SearchResult(topic=topic, keywords=[topic], issues=[], search_time=search_time),
            "analysis": None
        }

    def _create_error_result(self, topic: str, search_time: float, error: str) -> Dict[str, Any]:
        result = self._create_empty_result(topic, search_time)
        result["error"] = error
        return result

# ============= API 스키마 및 FastAPI 앱 (변경 없음) =============
class SearchRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=100, description="검색할 주제")
    time_period: str = Field(default="최근 1주일", description="검색 기간")
    analyze: bool = Field(default=True, description="분석 수행 여부")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("애플리케이션 시작")

    # API 키 상태 확인
    settings = get_settings()
    logger.info(f"Perplexity API Key: {'Configured' if settings.perplexity_api_key else 'Not configured'}")
    logger.info(f"Anthropic API Key: {'Configured' if settings.anthropic_api_key else 'Not configured'}")

    app.state.search_service = IssueSearchService()
    yield
    logger.info("애플리케이션 종료")
    await HTTPClientManager.close()

app = FastAPI(
    title="Hybrid AI Issue Search API",
    version="3.1.3", # 버전 업데이트 (콘텐츠 추출 개선)
    description="Perplexity의 구조화된 검색과 Claude의 심층 분석을 결합한 하이브리드 AI 기반 이슈 검색 및 분석 시스템",
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
    return {"message": "Hybrid AI Issue Search API v3.1.3"}

@app.post("/api/search")
async def search_issues_endpoint(request: Request, search_request: SearchRequest):
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
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

def format_final_report(result: Dict[str, Any], topic: str) -> Dict[str, Any]:
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

    summarized_contents = []
    source_links = []
    seen_urls = set()

    for issue in issues:
        if issue.is_valid_summary() and issue.url not in seen_urls:
            summary_text = issue.summary.strip()
            summarized_contents.append(f"### {issue.title}\n\n{summary_text}")

        if issue.url and issue.url not in seen_urls:
            source_links.append({"title": issue.title, "url": issue.url})
            seen_urls.add(issue.url)

    if analysis and analysis.summary:
        title = analysis.summary
    else:
        title = f"'{topic}'에 대한 최신 동향 분석"

    tags = [kw for kw in keywords if kw.lower() != topic.lower()]
    tags.insert(0, topic)
    tags = list(dict.fromkeys(tags))[:get_settings().keywords_for_tags]

    report_content = {
        "정리된 내용": "\n\n---\n\n".join(summarized_contents) if summarized_contents else "유효한 요약 내용이 없습니다. 출처 링크를 직접 확인해주세요.",
        "AI가 제공하는 리포트": analysis.to_full_text() if analysis else "종합 분석을 생성하지 못했습니다.",
        "출처 링크": source_links
    }

    if os.getenv('DEBUG_MODE', '').lower() == 'true':
        report_content["통계"] = {
            "검색된 이슈": len(issues),
            "유효한 요약": len(summarized_contents),
            "검색 시간": f"{search_result.search_time:.2f}초"
        }

    return {"제목": title, "태그": tags, "보고서": report_content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)