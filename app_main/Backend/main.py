"""
Improved Issue Search System with Perplexity and Claude
개선된 이슈 검색 시스템
"""

import asyncio
import re
import time
import json
from typing import List, Dict, Optional, Any, Protocol, TypedDict
from dataclasses import dataclass, field, asdict
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
    retry_delay: int = 2  # 1초에서 2초로 증가
    max_concurrent_requests: int = 3  # 5에서 3으로 감소

    # Search Settings
    default_time_period: str = "최근 1주일"
    min_issues_required: int = 3
    max_issues_per_search: int = 10
    keywords_per_retry: int = 10

    # Response Settings
    min_content_length: int = 100
    min_summary_length: int = 50
    max_tokens_keyword: int = 500
    max_tokens_search: int = 5000
    max_tokens_summary: int = 5000
    max_tokens_analysis: int = 1500

    @classmethod
    def from_env(cls) -> 'Settings':
        """환경 변수에서 설정 로드"""
        return cls(
            perplexity_api_key=os.getenv('PERPLEXITY_API_KEY'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            # 다른 환경 변수들도 필요하면 추가
            max_retries=int(os.getenv('MAX_RETRIES', '3')),
            timeout=int(os.getenv('TIMEOUT', '60')),
            retry_delay=int(os.getenv('RETRY_DELAY', '1')),
            max_concurrent_requests=int(os.getenv('MAX_CONCURRENT_REQUESTS', '5')),
            default_time_period=os.getenv('DEFAULT_TIME_PERIOD', '최근 1주일'),
            min_issues_required=int(os.getenv('MIN_ISSUES_REQUIRED', '3')),
            max_issues_per_search=int(os.getenv('MAX_ISSUES_PER_SEARCH', '10')),
            keywords_per_retry=int(os.getenv('KEYWORDS_PER_RETRY', '10')),
            min_content_length=int(os.getenv('MIN_CONTENT_LENGTH', '100')),
            min_summary_length=int(os.getenv('MIN_SUMMARY_LENGTH', '50')),
            max_tokens_keyword=int(os.getenv('MAX_TOKENS_KEYWORD', '500')),
            max_tokens_search=int(os.getenv('MAX_TOKENS_SEARCH', '5000')),
            max_tokens_summary=int(os.getenv('MAX_TOKENS_SUMMARY', '5000')),
            max_tokens_analysis=int(os.getenv('MAX_TOKENS_ANALYSIS', '1500')),
        )

@lru_cache()
def get_settings() -> Settings:
    """설정 싱글톤"""
    return Settings.from_env()

# ============= 로깅 설정 =============
def setup_logging():
    """로깅 설정 초기화"""
    logger.remove()

    # 콘솔 출력 (INFO 레벨)
    logger.add(
        sink=lambda msg: print(msg, end=''),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )

    # 파일 출력 (DEBUG 레벨) - 옵션
    if os.getenv('LOG_TO_FILE', '').lower() == 'true':
        logger.add(
            "logs/app_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="7 days",
            level="DEBUG",
            format="{time} | {level} | {name}:{function}:{line} - {message}",
            backtrace=True,
            diagnose=True
        )

setup_logging()

# ============= 커스텀 예외 =============
class IssueSearchError(Exception):
    """기본 예외 클래스"""
    pass

class APIError(IssueSearchError):
    """API 관련 예외"""
    pass

class ParsingError(IssueSearchError):
    """파싱 관련 예외"""
    pass

class ValidationError(IssueSearchError):
    """검증 관련 예외"""
    pass

# ============= 데이터 모델 =============
class IssueCategory(str, Enum):
    """이슈 카테고리"""
    NEWS = "뉴스"
    TECH = "기술"
    BUSINESS = "비즈니스"
    RESEARCH = "연구"
    POLICY = "정책"
    GENERAL = "일반"

@dataclass
class Issue:
    """이슈 데이터 모델"""
    title: str
    source: str
    published_date: Optional[str] = None
    category: str = IssueCategory.GENERAL
    url: Optional[str] = None
    content: Optional[str] = None
    summary: Optional[str] = None

    def is_valid_summary(self) -> bool:
        """유효한 요약인지 확인"""
        return bool(
            self.summary and
            len(self.summary) >= get_settings().min_summary_length and
            not any(self.summary.startswith(prefix) for prefix in ["실패", "콘텐츠", "처리 중"])
        )

@dataclass
class SearchResult:
    """검색 결과 모델"""
    topic: str
    keywords: List[str]
    issues: List[Issue]
    search_time: float
    total_found: int = field(init=False)

    def __post_init__(self):
        self.total_found = len(self.issues)

@dataclass
class AnalysisResult:
    """분석 결과 모델"""
    summary: str
    insights: List[str] = field(default_factory=list)
    future_outlook: Optional[str] = None
    analyzed_count: int = 0

    def to_full_text(self) -> str:
        """전체 분석 텍스트 생성"""
        parts = [self.summary, "\n주요 인사이트:"]
        parts.extend(f"{i}. {insight}" for i, insight in enumerate(self.insights, 1))
        if self.future_outlook:
            parts.append(f"\n향후 전망: {self.future_outlook}")
        return "\n".join(parts)

# ============= 프로토콜(인터페이스) 정의 =============
class KeywordGenerator(Protocol):
    """키워드 생성기 프로토콜"""
    async def generate_keywords(self, topic: str, max_keywords: int = 10) -> List[str]: ...

class ContentExtractor(Protocol):
    """콘텐츠 추출기 프로토콜"""
    async def extract_content(self, url: str) -> Optional[str]: ...

class Summarizer(Protocol):
    """요약기 프로토콜"""
    async def summarize_content(self, content: str, topic: str) -> str: ...

# ============= HTTP 클라이언트 관리 =============
class HTTPClientManager:
    """HTTP 클라이언트 재사용 관리"""
    _client: Optional[httpx.AsyncClient] = None

    @classmethod
    async def get_client(cls) -> httpx.AsyncClient:
        if cls._client is None:
            cls._client = httpx.AsyncClient(
                timeout=httpx.Timeout(get_settings().timeout),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
            )
        return cls._client

    @classmethod
    async def close(cls):
        if cls._client:
            await cls._client.aclose()
            cls._client = None

# ============= 유틸리티 함수 =============
class TextParser:
    """텍스트 파싱 유틸리티"""

    @staticmethod
    def extract_field(text: str, field_name: str) -> Optional[str]:
        """필드 값 추출"""
        pattern = rf'\*\*{field_name}\*\*:\s*(.*?)(?=\n\*\*|\Z)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def extract_url(text: str) -> Optional[str]:
        """URL 추출"""
        url_match = re.search(r'https?://[^\s/$.#].[^\s]*', text)
        return url_match.group(0).strip() if url_match else None

    @staticmethod
    def clean_keywords(content: str, topic: str) -> List[str]:
        """키워드 정리"""
        keywords = content.strip().split('\n')
        cleaned = [re.sub(r'^\s*[-\d.]\s*', '', kw).strip() for kw in keywords if kw.strip()]

        # 중복 제거하면서 순서 유지
        seen = set()
        unique = []
        for kw in cleaned:
            if kw and kw not in seen:
                seen.add(kw)
                unique.append(kw)

        # 주제가 없으면 추가
        if topic not in unique:
            unique.insert(0, topic)

        return unique

# ============= Perplexity 클라이언트 =============
class PerplexityClient:
    """개선된 Perplexity API 클라이언트"""

    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.perplexity_api_key
        self.base_url = settings.perplexity_base_url
        self.model = settings.perplexity_model
        self.parser = TextParser()

    async def _make_request(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.3,
        system_message: str = "당신은 정확한 정보 분석 전문가입니다."
    ) -> Dict[str, Any]:
        """API 요청 실행"""
        if not self.api_key:
            logger.warning("Perplexity API 키 없음, 기본 동작으로 전환")
            return {"choices": [{"message": {"content": ""}}]}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            client = await HTTPClientManager.get_client()
            response = await client.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Perplexity API HTTP 에러: {e.response.status_code}")
            raise APIError(f"API 요청 실패: {e}")
        except Exception as e:
            logger.error(f"Perplexity API 요청 실패: {e}")
            raise APIError(f"API 요청 중 오류: {e}")

    async def generate_keywords(self, topic: str, max_keywords: int = 10) -> List[str]:
        """향상된 키워드 생성"""
        if not self.api_key:
            return self._generate_fallback_keywords(topic, max_keywords)

        try:
            prompt = self._build_keyword_prompt(topic, max_keywords)
            response = await self._make_request(
                prompt,
                max_tokens=get_settings().max_tokens_keyword,
                temperature=0.8,
                system_message="당신은 최신 트렌드를 반영하여 검색 키워드를 생성하는 전문가입니다."
            )

            content = response['choices'][0]['message']['content']
            keywords = self.parser.clean_keywords(content, topic)
            return keywords[:max_keywords]

        except Exception as e:
            logger.error(f"키워드 생성 실패: {e}")
            return self._generate_fallback_keywords(topic, max_keywords)

    async def search_issues(self, keywords: List[str], time_period: str = "최근 1주일") -> List[Issue]:
        """향상된 이슈 검색"""
        if not self.api_key:
            return []

        prompt = self._build_search_prompt(keywords, time_period)

        try:
            response = await self._make_request(
                prompt,
                max_tokens=get_settings().max_tokens_search,
                temperature=0.3
            )

            content = response['choices'][0]['message']['content']
            return self._parse_issues(content)

        except Exception as e:
            logger.error(f"이슈 검색 실패: {e}")
            return []

    async def extract_content(self, url: str) -> Optional[str]:
        """URL에서 콘텐츠 추출"""
        if not self.api_key:
            return None

        prompt = f"다음 URL의 웹페이지에서 본문 텍스트만 추출해주세요:\nURL: {url}"

        try:
            response = await self._make_request(
                prompt,
                max_tokens=3000,
                temperature=0.1,
                system_message="당신은 웹페이지에서 본문을 정확하게 추출하는 도구입니다."
            )

            content = response['choices'][0]['message']['content'].strip()
            return content if len(content) >= get_settings().min_content_length else None

        except Exception as e:
            logger.error(f"콘텐츠 추출 실패 ({url}): {e}")
            return None

    def _build_keyword_prompt(self, topic: str, max_keywords: int) -> str:
        """키워드 생성 프롬프트"""
        base_prompt = f'주제: "{topic}"\n\n이 주제와 관련된 검색 키워드를 {max_keywords}개 생성해주세요.\n'

        if max_keywords <= 10:
            return base_prompt + """
- 핵심 키워드 3-4개
- 연관 키워드 3-4개  
- 최신 트렌드 키워드 2-3개

한 줄에 하나씩, 키워드만 나열해주세요."""
        else:
            return base_prompt + f"""
카테고리별로 골고루 생성:
1. 핵심 키워드 ({max_keywords//5}개)
2. 확장 키워드 ({max_keywords//5}개)
3. 최신 트렌드 ({max_keywords//5}개)
4. 기술/산업 용어 ({max_keywords//5}개)
5. 이슈/뉴스 키워드 (나머지)

한 줄에 하나씩, 중복 없이 다양하게."""

    def _build_search_prompt(self, keywords: List[str], time_period: str) -> str:
        """검색 프롬프트 구성"""
        keyword_groups = [keywords[i:i+5] for i in range(0, len(keywords), 5)]
        groups_text = '\n'.join(f"그룹 {i+1}: {', '.join(group)}" for i, group in enumerate(keyword_groups))

        return f"""키워드: {groups_text}
기간: {time_period}

위 키워드와 관련된 주요 이슈를 찾아주세요:
- 신뢰도 높은 출처 우선
- 각 그룹에서 최소 1개 이슈
- 중복 없이 다양하게

형식:
## **[제목]**
**출처**: [https://로 시작하는 완전한 URL]
**발행일**: [YYYY-MM-DD]
**카테고리**: [뉴스/기술/비즈니스/연구/정책]

최대 {get_settings().max_issues_per_search}개까지."""

    def _parse_issues(self, content: str) -> List[Issue]:
        """응답에서 이슈 파싱"""
        issues = []
        issue_blocks = re.finditer(r'(?s)(##\s*\*.*?(?=\n##\s*\*|\Z))', content)

        for match in issue_blocks:
            try:
                section = match.group(1).strip()
                issue = self._parse_issue_section(section)
                if issue and issue.url:
                    issues.append(issue)
            except Exception as e:
                logger.debug(f"이슈 파싱 스킵: {e}")
                continue

        return issues

    def _parse_issue_section(self, section: str) -> Optional[Issue]:
        """개별 이슈 섹션 파싱"""
        title_match = re.search(r'##\s*\*\*(.*?)\*\*\s*', section)
        if not title_match:
            return None

        title = title_match.group(1).strip()
        source = self.parser.extract_field(section, '출처') or 'Unknown'
        url = self.parser.extract_url(source)

        if not url:
            return None

        return Issue(
            title=title,
            source=source,
            url=url,
            published_date=self.parser.extract_field(section, '발행일'),
            category=self.parser.extract_field(section, '카테고리') or IssueCategory.GENERAL
        )

    def _generate_fallback_keywords(self, topic: str, max_keywords: int) -> List[str]:
        """폴백 키워드 생성"""
        templates = [
            topic, f"{topic} 최신", f"{topic} 뉴스", f"{topic} 동향",
            f"{topic} 분석", f"{topic} 기술", f"{topic} 트렌드",
            f"{topic} 2025", f"{topic} 산업", f"{topic} 활용",
            f"{topic} 사례", f"{topic} 전망", f"{topic} 이슈",
            f"{topic} 발전", f"{topic} 혁신", f"{topic} 연구"
        ]
        return templates[:max_keywords]

# ============= Claude 클라이언트 =============
class ClaudeClient:
    """개선된 Claude API 클라이언트"""

    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.anthropic_api_key
        self.model = settings.claude_model
        self.client = None
        if self.api_key:
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

    async def summarize_content(self, content: str, topic: str) -> str:
        """콘텐츠 요약"""
        if not self.client:
            return "Claude API 키가 없어 요약할 수 없습니다."

        # 콘텐츠 유효성 검사 강화
        clean_content = re.sub(r'<[^>]+>', '', content)
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()

        if len(clean_content) < get_settings().min_content_length:
            return "콘텐츠가 충분하지 않아 요약할 수 없습니다."

        # 추출 실패 메시지 패턴 확인
        error_patterns = [
            "웹페이지 접근 오류", "액세스할 수 없습니다", "내용을 찾을 수 없습니다",
            "페이지를 로드할 수 없습니다", "추출할 수 없습니다", "본문을 찾을 수 없습니다"
        ]
        if any(pattern in clean_content for pattern in error_patterns):
            logger.warning(f"콘텐츠 추출 실패로 추정되어 요약 건너뜀: {clean_content[:100]}...")
            return "제공된 텍스트에 요약할 내용이 없습니다."


        prompt = f"""주제: "{topic}"

다음 기사의 핵심 내용을 한국어로 요약해주세요.

요약 규칙:
1. 기사의 주요 사실과 정보를 정확하게 전달
2. 3-5개 문단으로 구성
3. 각 문단은 완전한 문장으로 끝나야 함
4. 추측이나 해석 없이 기사 내용만 요약
5. "죄송합니다", "제공된 본문", "요약할 수 없습니다" 등의 메타 설명 금지

--- 기사 본문 ---
{clean_content[:6000]}"""  # 토큰 제한

        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=get_settings().max_tokens_summary,
                temperature=0.3,
                system="당신은 뉴스 기사를 정확하고 간결하게 요약하는 전문가입니다. 기사 내용만 요약하고 다른 설명은 하지 마세요.",
                messages=[{"role": "user", "content": prompt}]
            )

            summary = message.content[0].text.strip()

            # 유효성 검증
            if self._is_invalid_summary(summary):
                logger.warning(f"유효하지 않은 요약 생성됨: {summary[:50]}...")
                return "요약 생성에 실패했습니다."

            # 마지막 문장이 완전한지 확인
            if summary and summary[-1] not in '.!?':
                last_period = summary.rfind('.')
                if last_period > 0:
                    summary = summary[:last_period + 1]

            return summary

        except Exception as e:
            logger.error(f"Claude 요약 실패: {e}")
            return "요약 생성 중 오류가 발생했습니다."

    async def analyze_issues(self, issues: List[Issue], topic: str) -> Optional[AnalysisResult]:
        """이슈 종합 분석"""
        if not self.client:
            return None

        valid_issues = [issue for issue in issues if issue.is_valid_summary()]
        if not valid_issues:
            return None

        issues_text = self._format_issues_for_analysis(valid_issues[:10])  # 최대 10개만
        prompt = f"""주제: "{topic}"

이슈 요약:
{issues_text}

위 이슈들을 분석하여 JSON 형식으로 응답:
{{
    "trend_summary": "전체 트렌드 1문장 요약",
    "insights": [
        "주요 인사이트 1 (2-3문장)",
        "주요 인사이트 2 (2-3문장)",
        "주요 인사이트 3 (2-3문장)"
    ],
    "future_outlook": "향후 전망 (1-2문장)"
}}"""

        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=get_settings().max_tokens_analysis,
                temperature=0.5,
                system="당신은 기술 트렌드 분석 전문가입니다. JSON 형식으로만 응답합니다.",
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text.strip()
            return self._parse_analysis_response(response_text, len(valid_issues))

        except Exception as e:
            logger.error(f"Claude 분석 실패: {e}")
            return None

    def _is_invalid_summary(self, summary: str) -> bool:
        """무효한 요약 체크"""
        invalid_patterns = [
            "내용이 없습니다",
            "죄송합니다",
            "콘텐츠",
            "제공된 본문이",
            "요약할 수 없습니다",
            "웹 크롤링",
            "기사가 아닌"
        ]
        return any(pattern in summary for pattern in invalid_patterns) or len(summary) < get_settings().min_summary_length

    def _format_issues_for_analysis(self, issues: List[Issue]) -> str:
        """분석용 이슈 포맷"""
        return "\n\n".join(
            f"[{i+1}] {issue.title}\n{issue.summary[:200]}..."
            for i, issue in enumerate(issues)
        )

    def _parse_analysis_response(self, response: str, analyzed_count: int) -> AnalysisResult:
        """분석 응답 파싱"""
        try:
            # JSON 마크다운 블록 정리
            if response.strip().startswith("```json"):
                response = response.strip()[7:-3].strip()

            data = json.loads(response)
            return AnalysisResult(
                summary=data.get('trend_summary', '분석 결과 없음'),
                insights=data.get('insights', []),
                future_outlook=data.get('future_outlook'),
                analyzed_count=analyzed_count
            )
        except (json.JSONDecodeError, IndexError):
            # JSON 파싱 실패 시 텍스트 분석
            logger.warning(f"JSON 파싱 실패, 텍스트로 대체: {response[:100]}...")
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            return AnalysisResult(
                summary=lines[0] if lines else "분석 결과 없음",
                insights=lines[1:4] if len(lines) > 1 else [],
                analyzed_count=analyzed_count
            )


# ============= 이슈 검색 서비스 =============
class IssueSearchService:
    """통합 이슈 검색 서비스"""

    def __init__(self):
        self.perplexity = PerplexityClient()
        self.claude = ClaudeClient()
        self.settings = get_settings()
        self.semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)
        self._request_count = 0
        self._last_request_time = time.time()

    async def search(
        self,
        topic: str,
        time_period: str = None,
        analyze: bool = True
    ) -> Dict[str, Any]:
        """이슈 검색 및 분석 메인 메서드"""
        start_time = time.time()
        time_period = time_period or self.settings.default_time_period

        try:
            # 1단계: 이슈 수집
            logger.info(f"'{topic}' 주제로 이슈 검색 시작")
            issues = await self._collect_issues_with_retry(topic, time_period)

            if not issues:
                logger.warning("이슈를 찾지 못했습니다")
                return self._create_empty_result(topic, time.time() - start_time)

            # 2단계: 콘텐츠 보강
            logger.info(f"{len(issues)}개 이슈 콘텐츠 보강 중...")
            await self._enrich_issues_content(issues, topic)

            # 3단계: 분석 수행
            analysis = None
            if analyze and self._has_valid_summaries(issues):
                logger.info("종합 분석 수행 중...")
                analysis = await self.claude.analyze_issues(issues, topic)

            # 결과 생성
            search_time = time.time() - start_time
            logger.info(f"검색 완료: {len(issues)}개 이슈, {search_time:.2f}초 소요")

            search_result = SearchResult(
                topic=topic,
                keywords=await self._get_used_keywords(topic),
                issues=issues,
                search_time=search_time
            )

            return {
                "search_result": search_result,  # SearchResult 객체 그대로 반환
                "analysis": analysis
            }

        except Exception as e:
            logger.error(f"검색 프로세스 오류: {e}")
            return self._create_error_result(topic, time.time() - start_time, str(e))

    async def _collect_issues_with_retry(self, topic: str, time_period: str) -> List[Issue]:
        """재시도 로직을 포함한 이슈 수집"""
        all_issues = []
        tried_keywords = set()

        for attempt in range(self.settings.max_retries):
            # API 속도 제한 대응
            await self._rate_limit_delay()

            # 점진적으로 더 많은 키워드 생성
            num_keywords = self.settings.keywords_per_retry * (attempt + 1)
            logger.info(f"키워드 생성 (시도 {attempt + 1}/{self.settings.max_retries})")

            keywords = await self.perplexity.generate_keywords(topic, num_keywords)
            new_keywords = [k for k in keywords if k not in tried_keywords]

            if not new_keywords:
                logger.warning("새로운 키워드가 없습니다")
                if attempt == 0:
                    new_keywords = keywords
                else:
                    continue

            tried_keywords.update(new_keywords)
            logger.info(f"검색 키워드 ({len(new_keywords)}개): {', '.join(new_keywords[:5])}...")

            # 이슈 검색
            issues = await self.perplexity.search_issues(new_keywords, time_period)

            # 중복 제거 및 추가
            if issues:
                existing_urls = {issue.url for issue in all_issues}
                new_issues = [issue for issue in issues if issue.url and issue.url not in existing_urls]

                if new_issues:
                    all_issues.extend(new_issues)
                    logger.info(f"새로운 이슈 {len(new_issues)}개 추가 (총 {len(all_issues)}개)")
                else:
                    logger.warning("모든 이슈가 중복됨")

                if len(all_issues) >= self.settings.min_issues_required:
                    logger.info(f"충분한 이슈 수집 완료: {len(all_issues)}개")
                    break
            else:
                logger.warning(f"이슈를 찾지 못함 (시도 {attempt + 1})")

            # 재시도 전 대기
            if attempt < self.settings.max_retries - 1:
                wait_time = self.settings.retry_delay * (attempt + 1)
                logger.info(f"{wait_time}초 대기 후 재시도...")
                await asyncio.sleep(wait_time)

        # 중복 제거 최종 확인
        unique_issues = []
        seen_urls = set()
        for issue in all_issues:
            if issue.url and issue.url not in seen_urls:
                unique_issues.append(issue)
                seen_urls.add(issue.url)

        logger.info(f"최종 수집 결과: {len(unique_issues)}개의 고유한 이슈")
        return unique_issues

    async def _rate_limit_delay(self):
        """API 속도 제한을 위한 지연"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        # 최소 0.5초 간격 유지
        if time_since_last < 0.5:
            await asyncio.sleep(0.5 - time_since_last)

        self._last_request_time = time.time()
        self._request_count += 1

    async def _enrich_issues_content(self, issues: List[Issue], topic: str):
        """이슈 콘텐츠 보강 (요약 생성)"""
        tasks = []
        for issue in issues:
            if issue.url and not issue.summary:
                task = self._enrich_single_issue(issue, topic)
                tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"이슈 보강 실패: {result}")

    async def _enrich_single_issue(self, issue: Issue, topic: str):
        """단일 이슈 콘텐츠 보강"""
        async with self.semaphore:  # 동시 요청 제한
            try:
                # 콘텐츠 추출
                if not issue.content:
                    content = await self.perplexity.extract_content(issue.url)
                    if content:
                        issue.content = content

                # 요약 생성
                if issue.content and not issue.summary:
                    summary = await self.claude.summarize_content(issue.content, topic)
                    issue.summary = summary

            except Exception as e:
                logger.error(f"이슈 보강 오류 ({issue.url}): {e}")
                issue.summary = "콘텐츠 처리 중 오류가 발생했습니다."

    async def _get_used_keywords(self, topic: str) -> List[str]:
        """사용된 키워드 목록"""
        try:
            return await self.perplexity.generate_keywords(topic, 10)
        except:
            return [topic]

    def _has_valid_summaries(self, issues: List[Issue]) -> bool:
        """유효한 요약이 있는지 확인"""
        return any(issue.is_valid_summary() for issue in issues)

    def _create_empty_result(self, topic: str, search_time: float) -> Dict[str, Any]:
        """빈 결과 생성"""
        search_result = SearchResult(
            topic=topic,
            keywords=[topic],
            issues=[],
            search_time=search_time
        )
        return {
            "search_result": search_result,
            "analysis": None
        }

    def _create_error_result(self, topic: str, search_time: float, error: str) -> Dict[str, Any]:
        """에러 결과 생성"""
        result = self._create_empty_result(topic, search_time)
        result["error"] = error
        return result

# ============= API 스키마 =============
class SearchRequest(BaseModel):
    """검색 요청 스키마"""
    topic: str = Field(..., min_length=1, max_length=100, description="검색할 주제")
    time_period: str = Field(default="최근 1주일", description="검색 기간")
    analyze: bool = Field(default=True, description="분석 수행 여부")

class ReportResponse(BaseModel):
    """최종 보고서 응답 스키마"""
    제목: str
    태그: List[str]
    보고서: Dict[str, Any]

# ============= FastAPI 앱 =============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 라이프사이클 관리"""
    # 시작
    logger.info("애플리케이션 시작")
    app.state.search_service = IssueSearchService()
    yield
    # 종료
    logger.info("애플리케이션 종료")
    await HTTPClientManager.close()

# FastAPI 앱 생성
app = FastAPI(
    title="Sejong_Dx_Hackathon Backend API",
    version="2.0.0",
    description="AI 기반 이슈 검색 및 분석 시스템",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 구체적인 도메인으로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= API 엔드포인트 =============
@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Sejong_Dx_Hackathon Backend API",
        "version": "2.0.0",
        "endpoints": {
            "search": "/api/search",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "perplexity_configured": bool(get_settings().perplexity_api_key),
            "claude_configured": bool(get_settings().anthropic_api_key)
        }
    }

@app.post("/api/search")
async def search_issues(request: Request, search_request: SearchRequest):
    """이슈 검색 및 분석 API"""
    try:
        # 검색 서비스 실행
        search_service: IssueSearchService = request.app.state.search_service
        result = await search_service.search(
            topic=search_request.topic,
            time_period=search_request.time_period,
            analyze=search_request.analyze
        )

        # 결과 포맷팅
        formatted_report = format_final_report(result, search_request.topic)
        return formatted_report

    except ValidationError as e:
        logger.error(f"검증 오류: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"검색 처리 오류: {type(e).__name__}: {e}")
        logger.exception("상세 에러:")  # 스택 트레이스 출력

        # 기본 응답 반환
        error_response = {
            "제목": f"'{search_request.topic}' 검색 중 오류 발생",
            "태그": [search_request.topic],
            "보고서": {
                "정리된 내용": "검색 처리 중 오류가 발생했습니다.",
                "AI가 제공하는 리포트": f"오류 유형: {type(e).__name__}\n상세: {str(e)}",
                "출처 링크": []
            }
        }
        return JSONResponse(content=error_response, status_code=200)  # 클라이언트가 처리할 수 있도록 200 반환

def format_final_report(result: Dict[str, Any], topic: str) -> Dict[str, Any]:
    """최종 보고서 포맷 생성"""
    search_result = result.get("search_result")
    analysis = result.get("analysis")

    # search_result가 None이거나 issues가 없는 경우
    if not search_result:
        return {
            "제목": f"'{topic}'에 대한 검색 결과가 없습니다",
            "태그": [topic],
            "보고서": {
                "정리된 내용": "검색 결과가 없습니다.",
                "AI가 제공하는 리포트": "분석할 내용이 없습니다.",
                "출처 링크": []
            }
        }

    # SearchResult 객체의 속성에 안전하게 접근
    issues = search_result.issues if hasattr(search_result, 'issues') else []
    keywords = search_result.keywords if hasattr(search_result, 'keywords') else []
    search_time = search_result.search_time if hasattr(search_result, 'search_time') else 0

    if not issues:
        return {
            "제목": f"'{topic}'에 대한 검색 결과가 없습니다",
            "태그": [topic],
            "보고서": {
                "정리된 내용": "검색 결과가 없습니다.",
                "AI가 제공하는 리포트": "분석할 내용이 없습니다.",
                "출처 링크": []
            }
        }

    # 요약 콘텐츠 생성 (유효한 요약만 포함)
    summarized_contents = []
    source_links = []
    seen_urls = set()  # 중복 URL 방지

    for issue in issues:
        # 유효한 요약만 포함
        if hasattr(issue, 'is_valid_summary') and issue.is_valid_summary():
            # 요약 텍스트 정리 (잘린 문장 제거)
            summary_text = issue.summary.strip()
            # 마지막 문장이 완전하지 않으면 제거
            if summary_text and not summary_text[-1] in '.!?':
                last_period = summary_text.rfind('.')
                if last_period > 0:
                    summary_text = summary_text[:last_period + 1]

            if summary_text:
                summarized_contents.append(f"### {issue.title}\n\n{summary_text}")

        # 중복되지 않은 URL만 추가
        if hasattr(issue, 'url') and issue.url and issue.url not in seen_urls:
            source_links.append(issue.url)
            seen_urls.add(issue.url)

    # 최종 보고서 구성
    if analysis and hasattr(analysis, 'summary'):
        title = analysis.summary
    else:
        title = f"{topic}에 대한 분석 요약"

    # 태그는 최대 5개까지
    tags = []
    if keywords:
        tags = [tag for tag in keywords[:5] if tag != topic]
    if topic not in tags:
        tags.insert(0, topic)
    tags = tags[:3]  # 최종적으로 3개만

    # 보고서 내용
    report_content = {
        "정리된 내용": "\n\n---\n\n".join(summarized_contents) if summarized_contents else "유효한 요약 내용이 없습니다.",
        "AI가 제공하는 리포트": analysis.to_full_text() if analysis and hasattr(analysis, 'to_full_text') else "분석 내용이 없습니다.",
        "출처 링크": source_links
    }

    # 통계 정보 추가 (선택사항 - 환경변수로 제어)
    if os.getenv('DEBUG_MODE', '').lower() == 'true':
        report_content["통계"] = {
            "검색된 이슈": len(issues),
            "유효한 요약": len(summarized_contents),
            "검색 시간": f"{search_time:.2f}초"
        }

    return {
        "제목": title,
        "태그": tags,
        "보고서": report_content
    }

# ============= 메인 실행 =============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
        }
    )