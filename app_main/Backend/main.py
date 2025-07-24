"""
Simple Issue Search System with Perplexity and Claude
Perplexity와 Claude를 사용한 간단한 이슈 검색 시스템
"""

import asyncio
import re
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import httpx
from loguru import logger
import anthropic
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 환경 변수 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI()

# CORS 미들웨어 추가 (개발 중 모든 출처 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 실제 프론트엔드 주소로 변경해야 합니다.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============= 데이터 모델 =============
@dataclass
class IssueItem:
    """검색된 이슈 아이템"""
    title: str
    summary: str
    source: str
    published_date: Optional[str]
    category: str
    url: Optional[str] = None
    content: Optional[str] = None


@dataclass
class SearchResult:
    """검색 결과"""
    topic: str
    keywords: List[str]
    issues: List[IssueItem]
    total_found: int
    search_time: float


# ============= Perplexity 키워드 생성 =============
class PerplexityKeywordGenerator:
    """Perplexity를 사용한 실시간 키워드 생성기"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "sonar-pro"

    async def generate_keywords(self, topic: str, max_keywords: int = 10) -> List[str]:
        """주제에 대한 키워드 실시간 생성"""
        if not self.api_key:
            logger.warning("Perplexity API 키가 없어 기본 키워드 사용")
            return self._generate_basic_keywords(topic)

        try:
            prompt = f'''
주제: "{topic}"

이 주제와 관련된 실시간 검색 키워드를 {max_keywords}개 생성해주세요.
- 가장 관련성 높은 핵심 키워드 3-4개
- 연관 검색어 또는 확장 키워드 3-4개
- 현재 시간 기준 최신 트렌드를 반영한 키워드 2-3개

한 줄에 하나씩, 키워드만 나열해주세요.
'''
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "당신은 최신 트렌드를 반영하여 검색 키워드를 생성하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.7,
            }

            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                api_response = response.json()

            content = api_response['choices'][0]['message']['content']
            keywords = content.strip().split('\n')
            keywords = [re.sub(r'^\s*-\s*', '', kw).strip() for kw in keywords if kw.strip()]

            if topic not in keywords:
                keywords.insert(0, topic)

            return keywords

        except Exception as e:
            logger.error(f"Perplexity 키워드 생성 실패: {e}")
            return self._generate_basic_keywords(topic)

    def _generate_basic_keywords(self, topic: str) -> List[str]:
        """기본 키워드 생성 (API 없을 때)"""
        return [
            topic,
            f"{topic} 최신",
            f"{topic} 뉴스",
            f"{topic} 동향",
            f"{topic} 분석"
        ]


# ============= Perplexity 클라이언트 =============
class SimplePerplexityClient:
    """간단한 Perplexity API 클라이언트"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "sonar-pro"

    async def search_issues(self, keywords: List[str], time_period: str = "최근 1주일") -> Dict[str, Any]:
        """이슈 검색 (URL 위주)"""
        prompt = f'''
'{ ", ".join(keywords)}' 키워드와 관련하여 '{time_period}' 동안 발행된 주요 이슈를 찾아주세요.
**중요: 블로그나 개인 의견보다는 공식 발표, 신뢰도 높은 뉴스 매체의 정보를 우선으로 찾아주세요.**

각 이슈마다 다음 형식으로 작성해주세요:

## **[이슈 제목]**
**출처**: [전체 URL]
**발행일**: [YYYY-MM-DD 형식]
**카테고리**: [뉴스/기술/비즈니스 등]

최대 5개의 가장 관련성 높은 이슈를 찾아주세요.
'''

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 정확한 정보 분석 전문가입니다. 신뢰도 높은 출처의 URL을 포함해 주세요."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.2
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    async def extract_content_from_url(self, url: str) -> Optional[str]:
        """URL에서 본문 콘텐츠 추출"""
        if not self.api_key:
            logger.warning("Perplexity API 키가 없어 콘텐츠를 추출할 수 없습니다.")
            return None

        prompt = f'''
다음 URL의 웹페이지에서 광고, 메뉴, 댓글 등 부가적인 요소를 제외하고 순수 본문 텍스트만 추출해주세요.
추출된 텍스트는 마크다운 형식이 아니어야 합니다. 순수 텍스트로만 제공해주세요.

URL: {url}
'''
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "당신은 웹페이지에서 본문을 정확하게 추출하는 도구입니다."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 3000,
            "temperature": 0.1,
        }

        try:
            async with httpx.AsyncClient(timeout=90) as client:
                response = await client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                api_response = response.json()
            return api_response['choices'][0]['message']['content'].strip()
        except Exception as e:
            logger.error(f"URL {url}에서 콘텐츠 추출 실패: {e}")
            return None


# ============= Claude를 사용한 향상된 분석 =============
class ClaudeAnalyzer:
    """Claude를 사용한 이슈 분석 및 요약"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key) if self.api_key else None
        self.model = "claude-3-5-sonnet-20240620" # 최신 모델로 변경

    async def summarize_issue(self, content: str, topic: str) -> str:
        """Claude를 사용하여 개별 이슈 콘텐츠 요약"""
        if not self.client:
            return "Claude API 키가 없어 요약할 수 없습니다."

        prompt = f'''
주제: "{topic}"

다음은 해당 주제와 관련된 기사 본문입니다. 이 본문을 한국어로 상세히 요약해주세요.
- 핵심 내용을 명확하게 전달해야 합니다.
- 원문의 주요 주장과 근거를 포함해야 합니다.
- 3-5 문단으로 구성된 상세한 요약문을 작성해주세요.

--- 기사 본문 ---
{content}
'''

        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.3,
                system="당신은 기사 내용을 정확하고 상세하게 요약하는 전문 에디터입니다.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            summary = message.content[0].text.strip()
            return summary
        except Exception as e:
            logger.error(f"Claude 요약 실패: {e}")
            return "콘텐츠 요약에 실패했습니다."

    async def analyze_issues(self, issues: List[IssueItem], topic: str) -> Dict[str, Any]:
        """수집된 이슈들을 분석하고 인사이트 도출"""
        if not self.client or not issues:
            return {"summary": "분석 불가", "insights": []}

        issues_text = "\n\n".join([
            f"제목: {issue.title}\n요약: {issue.summary}\n출처: {issue.source}\n날짜: {issue.published_date}"
            for issue in issues if issue.summary and "실패" not in issue.summary
        ])

        if not issues_text:
            return {"summary": "분석할 이슈가 없습니다.", "insights": []}

        prompt = f'''
주제: "{topic}"

다음은 관련 이슈들의 요약입니다:
{issues_text}

위 이슈들을 종합적으로 분석하여 다음을 제공해주세요:
1. 전체적인 트렌드 요약 (2-3문장)
2. 주요 인사이트 3가지 (각 인사이트는 2-3문장으로 설명)
3. 향후 예상되는 동향 (1-2문장)

간결하고 명확하게 작성해주세요.
'''

        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.5,
                system="당신은 기술 트렌드 분석 전문가입니다. 이슈들을 종합적으로 분석하여 인사이트를 도출합니다.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            analysis_text = message.content[0].text
            return {
                "summary": analysis_text.split("\n")[0] if analysis_text else "분석 결과 없음",
                "full_analysis": analysis_text,
                "analyzed_count": len([issue for issue in issues if issue.summary and "실패" not in issue.summary])
            }
        except Exception as e:
            logger.error(f"Claude 분석 실패: {e}")
            return {"summary": "분석 실패", "insights": []}


# ============= 통합 이슈 검색기 =============
class ClaudeIssueSearcher:
    """Perplexity 검색, Claude 요약 및 분석을 활용한 이슈 검색기"""

    def __init__(self, anthropic_key: Optional[str] = None, perplexity_key: Optional[str] = None):
        self.keyword_generator = PerplexityKeywordGenerator(perplexity_key)
        self.perplexity_client = SimplePerplexityClient(perplexity_key)
        self.analyzer = ClaudeAnalyzer(anthropic_key)

    async def search(self, topic: str, time_period: str = "최근 1주일", analyze: bool = True) -> Dict[str, Any]:
        """주제에 대한 이슈 검색, 요약 및 분석"""
        start_time = time.time()
        try:
            logger.info(f"Perplexity로 키워드 생성 중: {topic}")
            keywords = await self.keyword_generator.generate_keywords(topic)
            logger.info(f"생성된 키워드: {keywords}")

            logger.info("Perplexity로 이슈 검색 중...")
            api_response = await self.perplexity_client.search_issues(keywords, time_period)

            issues = self._parse_response(api_response, keywords)
            logger.info(f"{len(issues)}개 이슈 URL 발견. 상세 내용 요약 시작...")

            summary_tasks = []
            for issue in issues:
                if issue.url:
                    summary_tasks.append(self._fetch_and_summarize_issue(issue, topic))
            
            await asyncio.gather(*summary_tasks)

            analysis = None
            if analyze and any(issue.summary and "실패" not in issue.summary for issue in issues):
                logger.info("Claude로 전체 이슈 분석 중...")
                analysis = await self.analyzer.analyze_issues(issues, topic)

            search_time = time.time() - start_time
            logger.info(f"검색 및 요약 완료: {len(issues)}개 이슈 처리 ({search_time:.2f}초)")

            # dataclass를 dict로 변환
            result = SearchResult(topic=topic, keywords=keywords, issues=issues, total_found=len(issues), search_time=search_time)
            return {"search_result": asdict(result), "analysis": analysis}

        except Exception as e:
            logger.error(f"전체 검색 프로세스 실패: {e}")
            search_result = SearchResult(topic=topic, keywords=[topic], issues=[], total_found=0, search_time=time.time() - start_time)
            return {
                "search_result": asdict(search_result),
                "analysis": None,
                "error": str(e)
            }

    async def _fetch_and_summarize_issue(self, issue: IssueItem, topic: str):
        logger.info(f"콘텐츠 추출 중: {issue.url}")
        content = await self.perplexity_client.extract_content_from_url(issue.url)
        if content:
            issue.content = content
            logger.info(f"Claude로 '{issue.title}' 콘텐츠 요약 중...")
            summary = await self.analyzer.summarize_issue(content, topic)
            issue.summary = summary
        else:
            issue.summary = "콘텐츠를 추출하지 못했습니다."

    def _parse_response(self, api_response: Dict[str, Any], keywords: List[str]) -> List[IssueItem]:
        """API 응답 파싱 (URL 위주)"""
        try:
            content = api_response['choices'][0]['message']['content']
            issue_blocks = re.finditer(r'(?s)(##\s*\*.*?(?=\n##\s*\*|\Z))', content)
            issues = []
            for match in issue_blocks:
                section = match.group(1).strip()
                issue = self._parse_issue_section(section)
                if issue and issue.url:
                    issues.append(issue)
            return issues
        except Exception as e:
            logger.error(f"응답 파싱 실패: {e}")
            return []

    def _parse_issue_section(self, section: str) -> Optional[IssueItem]:
        """이슈 섹션 파싱 (URL 위주)"""
        try:
            title_match = re.search(r'##\s*\*\*(.*?)\*\*\s*', section)
            if not title_match: return None
            title = title_match.group(1).strip()

            source = self._extract_field(section, '출처') or 'Unknown'
            url_match = re.search(r'https?://[^\s/$.#].[^\s]*', source)
            url = url_match.group(0).strip() if url_match else None
            if not url: return None

            date_str = self._extract_field(section, '발행일')
            category = self._extract_field(section, '카테고리') or 'general'

            return IssueItem(title=title, summary="", source=source, published_date=date_str, category=category, url=url)
        except Exception as e:
            logger.error(f"이슈 섹션 파싱 실패: {e}")
            return None

    def _extract_field(self, text: str, field_name: str) -> Optional[str]:
        """필드 값 추출"""
        pattern = rf'\*\*{field_name}\*\*:\s*(.*?)(?=\n\*\*|\Z)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None


# ============= API 엔드포인트 =============
class SearchRequest(BaseModel):
    topic: str
    time_period: str = "최근 1주일"
    analyze: bool = True

@app.post("/api/search")
async def run_search(request: SearchRequest):
    """
    주제를 받아 이슈를 검색, 요약, 분석하고 결과를 JSON으로 반환합니다.
    """
    searcher = ClaudeIssueSearcher()
    topic = request.topic
    result = await searcher.search(topic, request.time_period, request.analyze)

    search_result_data = result.get("search_result")
    analysis = result.get("analysis")

    if search_result_data and search_result_data.get("issues"):
        # search_result_data는 이미 dict입니다。
        summarized_content = []
        for issue in search_result_data["issues"]:
            if issue.get("summary") and "실패" not in issue["summary"]:
                summarized_content.append(f"### {issue['title']}\n\n{issue['summary']}")

        final_report = {
            "제목": analysis.get("summary", f"{topic}에 대한 분석 요약") if analysis else f"{topic}에 대한 분석 요약",
            "태그": search_result_data.get("keywords", [])[:3],
            "보고서": {
                "정리된 내용": "\n\n---\n\n".join(summarized_content),
                "AI가 제공하는 리포트": analysis.get("full_analysis", "상세 분석 내용이 없습니다.") if analysis else "상세 분석 내용이 없습니다.",
                "출처 링크": [issue.get("url") for issue in search_result_data["issues"] if issue.get("url")]
            }
        }
        return JSONResponse(content=final_report)
    else:
        error_report = {
            "error": "검색 또는 분석에 실패했거나 결과가 없습니다.",
            "topic": topic,
            "details": result.get("error")
        }
        return JSONResponse(content=error_report, status_code=500)

@app.get("/")
async def root():
    return {"message": "Sejong_Dx_Hackathon Backend API"}

# uvicorn으로 서버를 실행하려면 터미널에 다음을 입력하세요:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000