"""
Claude와 Perplexity를 활용한 지능형 이슈 검색 시스템 (기간 지정 기능 개선 버전)
- 하이브리드 검색, LLM-as-a-judge 검증, AI 분석 기능 포함
"""

import asyncio
import re
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta  # 추가된 부분
import httpx
from loguru import logger
import anthropic
import os
from dotenv import load_dotenv
import sys

# .env 파일에서 환경 변수를 불러옵니다.
load_dotenv()


# --- 데이터 모델 정의 ---

@dataclass
class VerificationResult:
    """LLM 판사의 검증 결과를 담는 데이터 클래스입니다."""
    status: str = "미검증"  # "VERIFIED", "UNVERIFIED", "NEEDS_REVIEW" 중 하나의 상태를 가집니다.
    reasoning: str = "아직 검증이 수행되지 않았습니다."

@dataclass
class IssueItem:
    """개별 이슈 정보를 저장하는 데이터 클래스입니다."""
    title: str
    summary: str
    source: str
    published_date: Optional[str]
    relevance_score: float
    category: str
    url: Optional[str] = None
    content: Optional[str] = None
    verification: VerificationResult = field(default_factory=VerificationResult) # 이슈 검증 결과를 저장합니다.

@dataclass
class SearchResult:
    """최종 검색 결과를 종합하는 데이터 클래스입니다."""
    topic: str
    keywords: List[str]
    issues: List[IssueItem]
    initial_found: int
    verified_found: int
    search_time: float
    search_start_date: str  # 추가된 부분: 검색 시작일
    search_end_date: str    # 추가된 부분: 검색 종료일


# --- 키워드 생성기 (Claude Keyword Generator) ---
# 참고: 현재 하이브리드 방식에서는 직접 사용되지 않지만, 추후 확장을 위해 구조를 유지합니다.
class ClaudeKeywordGenerator:
    """Claude를 이용해 검색 키워드를 생성합니다."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key) if self.api_key else None
        self.model = "claude-3-opus-20240229"

    async def generate_keywords(self, topic: str, max_keywords: int = 10) -> List[str]:
        """주어진 주제에 대한 검색 키워드를 생성합니다."""
        if not self.client:
            logger.warning("Anthropic API 키가 설정되지 않아 기본 키워드 생성 로직을 사용합니다.")
            return self._generate_basic_keywords(topic)

        try:
            prompt = f"""주제: "{topic}"
이 주제와 관련된 검색 키워드를 {max_keywords}개 생성해주세요.
- 핵심 키워드 3-4개
- 관련 용어 3-4개
- 최신 트렌드 키워드 2-3개
한 줄에 하나씩, 키워드만 작성해주세요."""
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.7,
                system="당신은 특정 주제에 대해 깊이 있고 관련성 높은 검색어를 제안하는 키워드 전문가입니다.",
                messages=[{"role": "user", "content": prompt}]
            )
            keywords = [kw.strip() for kw in message.content[0].text.strip().split('\n') if kw.strip()]
            if topic not in keywords:
                keywords.insert(0, topic)
            return keywords[:max_keywords]
        except Exception as e:
            logger.error(f"Claude 키워드 생성 중 오류 발생: {e}")
            return self._generate_basic_keywords(topic)

    def _generate_basic_keywords(self, topic: str) -> List[str]:
        """API 사용이 불가능할 때, 기본적인 키워드를 생성합니다."""
        return [topic, f"{topic} 최신", f"{topic} 뉴스", f"{topic} 동향", f"{topic} 분석"]


# --- Perplexity 클라이언트 (검색 및 검증 기능) ---
class SimplePerplexityClient:
    """Perplexity API를 사용해 이슈를 검색하고, 각 이슈의 사실 여부를 검증합니다."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        if not self.api_key:
            raise ValueError("Perplexity API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "sonar-pro"

    # 변경된 부분: time_period 대신 start_date와 end_date를 받도록 수정
    async def search_issues(self, keywords: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """키워드를 기반으로 지정된 기간 동안의 관련 최신 이슈를 검색합니다."""
        prompt = f"""'{", ".join(keywords)}' 관련 이슈를 검색합니다.

    **중요: 날짜 제약 조건**
    - 반드시 {start_date}부터 {end_date}까지의 기간에 발행된 이슈만 포함하세요
    - 이 기간 외의 이슈는 절대 포함하지 마세요
    - 각 이슈의 발행일이 위 기간 내에 있는지 반드시 확인하세요

    오늘 날짜는 {datetime.now().strftime('%Y-%m-%d')}입니다.

    각 이슈는 아래 형식을 반드시 지켜 작성해주세요:
    ## **[이슈 제목]**
    **요약**: [간단한 요약]
    **출처**: [웹사이트명 또는 URL]
    **발행일**: [YYYY-MM-DD 형식] (반드시 {start_date} ~ {end_date} 사이여야 함)
    **카테고리**: [뉴스/기술/비즈니스 등]

    최소 3개, 최대 10개의 이슈를 찾아주세요. 날짜가 범위를 벗어난 이슈는 포함하지 마세요."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": f"당신은 정확한 정보 분석가입니다. 반드시 {start_date}부터 {end_date}까지의 기간에 발행된 이슈만 찾아야 합니다. 이 기간 외의 정보는 절대 포함하지 마세요. 각 이슈의 날짜가 지정된 범위 내에 있는지 반드시 확인하고, 사실에 기반한 출처와 정확한 날짜를 포함시켜주세요."
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4000,
            "temperature": 0.2
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    async def verify_issue(self, issue: IssueItem) -> VerificationResult:
        """LLM을 'judge'로 활용해, 각 이슈의 사실 여부를 검증합니다."""
        logger.debug(f"검증 시작: '{issue.title}'")

        prompt = f"""당신은 편견 없는 사실 확인 전문가(Fact-Checker)입니다. 아래 뉴스 이슈의 제목, 요약, 출처, 발행일이 일관되고 사실인지 웹 검색을 통해 검증해주세요.
    [검증할 이슈]
    - 제목: {issue.title}
    - 요약: {issue.summary}
    - 출처: {issue.source}
    - 발행일: {issue.published_date}
    [응답 형식]
    아래 형식을 반드시 지켜 한 단어의 판결과 한 문장의 이유를 제시하세요:
    VERDICT: [VERIFIED, UNVERIFIED, NEEDS_REVIEW 중 하나]
    REASONING: [판결에 대한 간결한 이유]"""

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an impartial fact-checker AI."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300, "temperature": 0.1
        }

        try:
            async with httpx.AsyncClient(timeout=45) as client:
                response = await client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content']

                # 응답 내용 로깅
                logger.debug(f"검증 API 응답 ('{issue.title}'): {content}")

                verdict_match = re.search(r"VERDICT:\s*(\w+)", content)
                reasoning_match = re.search(r"REASONING:\s*(.*)", content, re.DOTALL)

                status = verdict_match.group(1).strip() if verdict_match else "NEEDS_REVIEW"
                reasoning = reasoning_match.group(1).strip() if reasoning_match else "응답을 파싱할 수 없습니다."

                # 검증 결과 상세 로깅
                logger.info(f"검증 결과 - '{issue.title}': {status}")
                logger.debug(f"검증 이유: {reasoning}")

                return VerificationResult(status=status, reasoning=reasoning)

        except Exception as e:
            logger.error(f"이슈 검증 API 호출 중 오류: '{issue.title}', 오류: {e}")
            return VerificationResult(status="NEEDS_REVIEW", reasoning=f"검증 중 API 오류가 발생했습니다: {e}")


# --- 이슈 분석기 (Claude Analyzer) ---
class ClaudeAnalyzer:
    """Claude를 사용해 검증된 이슈들을 종합적으로 분석하고 인사이트를 도출합니다."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key) if self.api_key else None
        self.model = "claude-3-opus-20240229"

    async def analyze_issues(self, issues: List[IssueItem], topic: str) -> Dict[str, Any]:
        """검증된 이슈 목록을 받아 전체적인 트렌드를 요약하고 분석합니다."""
        if not self.client or not issues:
            return {"summary": "분석할 이슈가 없습니다.", "full_analysis": ""}

        issues_text = "\n\n".join([f"제목: {issue.title}\n요약: {issue.summary}" for issue in issues[:5]])
        prompt = f"""주제: "{topic}"
다음은 이 주제와 관련하여 검증된 최신 이슈 목록입니다:
{issues_text}
위 내용을 종합하여 다음 항목에 대해 분석해주세요:
1. 전체 트렌드 요약 (2-3 문장)
2. 주목할 만한 핵심 인사이트 3가지
3. 향후 전망 (1-2 문장)
결과는 명확하고 간결하게 작성해주세요."""
        try:
            message = await self.client.messages.create(
                model=self.model, max_tokens=1000, temperature=0.5,
                system="당신은 기술 트렌드 분석 전문가입니다. 여러 뉴스를 종합해 핵심적인 인사이트를 도출합니다.",
                messages=[{"role": "user", "content": prompt}]
            )
            analysis_text = message.content[0].text
            return {
                "summary": analysis_text.split("\n")[0],
                "full_analysis": analysis_text,
                "analyzed_count": len(issues[:5])
            }
        except Exception as e:
            logger.error(f"Claude 분석 중 오류 발생: {e}")
            return {"summary": "분석 중 오류가 발생했습니다.", "full_analysis": ""}


# --- 통합 이슈 검색기 (검색, 검증, 분석 파이프라인) ---
class ClaudeIssueSearcher:
    """이슈 검색, 검증, 분석의 전체 과정을 관리하는 메인 클래스입니다."""

    def __init__(self, anthropic_key: Optional[str] = None, perplexity_key: Optional[str] = None):
        self.keyword_generator = ClaudeKeywordGenerator(anthropic_key)
        self.perplexity_client = SimplePerplexityClient(perplexity_key)
        self.analyzer = ClaudeAnalyzer(anthropic_key)

    # 변경된 부분: time_period 대신 days_ago를 받도록 수정
    async def search(self, topic: str, days_ago: int = 7, analyze: bool = True) -> Dict[str, Any]:
        """주어진 주제에 대해 이슈를 검색, 검증하고, 선택적으로 분석까지 수행합니다."""
        start_time = time.time()
        initial_issues, verified_issues = [], []

        # 추가된 부분: 검색 기간 계산
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_ago)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        logger.info(f"검색 기간: {start_date_str} 부터 {end_date_str} 까지")

        try:
            # 1단계: 검색 키워드 생성
            logger.info(f"'{topic}'에 대한 검색 키워드를 생성합니다.")
            keywords = self.keyword_generator._generate_basic_keywords(topic)
            logger.info(f"생성된 키워드: {keywords}")

            # 2단계: Perplexity로 1차 이슈 검색 (변경된 부분: 날짜 인자 전달)
            logger.info("Perplexity를 통해 1차 이슈 검색을 시작합니다...")
            api_response = await self.perplexity_client.search_issues(keywords, start_date_str, end_date_str)

            # 3단계: 1차 검색 결과 파싱
            initial_issues = self._parse_response(api_response, keywords)
            if not initial_issues:
                raise ValueError("1차 검색 결과, 유효한 이슈를 찾지 못했습니다.")

            # 4단계: LLM-as-a-judge 통해 각 이슈의 사실 여부 검증 부분 수정
            logger.info(f"LLM-as-a-judge를 통해 {len(initial_issues)}개 이슈의 사실 여부를 검증합니다...")
            verification_tasks = [self.perplexity_client.verify_issue(issue) for issue in initial_issues]
            verification_results = await asyncio.gather(*verification_tasks)

            for issue, verification in zip(initial_issues, verification_results):
                issue.verification = verification
                if verification.status == "VERIFIED":
                    verified_issues.append(issue)
                    logger.success(f"✅ 검증 통과: '{issue.title}'")
                else:
                    # 실패/보류 시 더 상세한 로깅
                    logger.warning(f"❌ 검증 실패/보류: '{issue.title}'")
                    logger.warning(f"   상태: {verification.status}")
                    logger.warning(f"   이유: {verification.reasoning}")
                    logger.debug(f"   전체 이슈 정보: 제목={issue.title}, 출처={issue.source}, 날짜={issue.published_date}")

            # 5단계: (선택) 검증된 이슈들을 Claude로 심층 분석
            analysis = None
            if analyze and verified_issues:
                logger.info("Claude를 통해 검증된 이슈들의 심층 분석을 시작합니다...")
                analysis = await self.analyzer.analyze_issues(verified_issues, topic)

            # 6단계: 최종 결과 정리 및 반환 (변경된 부분: 날짜 정보 추가)
            search_time = time.time() - start_time
            logger.info(f"모든 과정 완료. (총 소요 시간: {search_time:.2f}초)")
            result = SearchResult(
                topic=topic, keywords=keywords, issues=verified_issues,
                initial_found=len(initial_issues), verified_found=len(verified_issues), search_time=search_time,
                search_start_date=start_date_str, search_end_date=end_date_str
            )
            return {"search_result": result, "analysis": analysis}

        except Exception as e:
            logger.error(f"전체 검색 파이프라인에서 오류 발생: {e}")
            return {
                "search_result": SearchResult(
                    topic=topic, keywords=[topic], issues=[],
                    initial_found=len(initial_issues), verified_found=0, search_time=time.time() - start_time,
                    search_start_date=start_date_str, search_end_date=end_date_str
                ),
                "analysis": None
            }

    def _parse_response(self, api_response: Dict[str, Any], keywords: List[str]) -> List[IssueItem]:
        """Perplexity의 검색 응답을 파싱하여 IssueItem 리스트로 변환합니다."""
        try:
            content = api_response['choices'][0]['message']['content']

            # 디버깅: 전체 응답 내용 출력
            logger.debug(f"Perplexity API 응답 전체 내용:\n{content}\n")

            # 더 유연한 정규식 패턴들 시도
            patterns = [
                r'(?s)(##\s*\*\*.*?(?=\n##\s*\*\*|\Z))',  # 원래 패턴
                r'(?s)(##\s*\[.*?\].*?(?=\n##\s*\[|\Z))',  # ## [제목] 형식
                r'(?s)(\d+\.\s*\*\*.*?(?=\n\d+\.\s*\*\*|\Z))',  # 1. **제목** 형식
                r'(?s)(\d+\.\s*.*?(?=\n\d+\.|\Z))',  # 1. 제목 형식
            ]

            issues = []
            issue_found = False

            for pattern in patterns:
                issue_blocks = list(re.finditer(pattern, content))
                if issue_blocks:
                    logger.info(f"패턴 '{pattern[:30]}...'로 {len(issue_blocks)}개의 이슈 블록을 찾았습니다.")
                    issue_found = True

                    for match in issue_blocks:
                        section = match.group(1).strip()
                        logger.debug(f"파싱 중인 섹션:\n{section[:200]}...")

                        if issue := self._parse_issue_section(section):
                            issue.relevance_score = self._calculate_relevance(issue, keywords)
                            issues.append(issue)
                            logger.info(f"이슈 파싱 성공: '{issue.title}'")
                    break

            if not issue_found:
                logger.warning("어떤 패턴으로도 이슈를 찾지 못했습니다. 응답 형식을 확인해주세요.")
                # 간단한 폴백 파싱 시도
                issues = self._fallback_parse(content, keywords)

            logger.info(f"총 {len(issues)}개의 이슈를 파싱했습니다.")
            return issues

        except (KeyError, IndexError, AttributeError) as e:
            logger.error(f"API 응답 파싱 중 예상치 못한 구조: {e}")
            logger.error(f"응답 구조: {api_response.keys() if isinstance(api_response, dict) else type(api_response)}")
            return []

    def _fallback_parse(self, content: str, keywords: List[str]) -> List[IssueItem]:
        """정규식 파싱이 실패할 경우 폴백 파싱 방법"""
        issues = []

        # 제목 찾기 시도
        title_patterns = [
            r'제목:\s*(.+?)(?:\n|$)',
            r'\*\*(.+?)\*\*',
            r'###?\s*(.+?)(?:\n|$)',
            r'\d+\.\s*(.+?)(?:\n|$)'
        ]

        lines = content.split('\n')
        current_issue = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 새로운 이슈 시작 감지
            for pattern in title_patterns:
                match = re.search(pattern, line)
                if match and not any(field in line.lower() for field in ['요약', '출처', '발행일', '카테고리']):
                    if current_issue.get('title'):
                        # 이전 이슈 저장
                        issue = self._create_issue_from_dict(current_issue)
                        if issue:
                            issue.relevance_score = self._calculate_relevance(issue, keywords)
                            issues.append(issue)

                    current_issue = {'title': match.group(1).strip()}
                    break

            # 필드 정보 추출
            field_patterns = {
                '요약': r'요약:\s*(.+?)(?:\n|$)',
                '출처': r'출처:\s*(.+?)(?:\n|$)',
                '발행일': r'발행일:\s*(\d{4}-\d{2}-\d{2})',
                '카테고리': r'카테고리:\s*(.+?)(?:\n|$)'
            }

            for field, pattern in field_patterns.items():
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    current_issue[field] = match.group(1).strip()

        # 마지막 이슈 저장
        if current_issue.get('title'):
            issue = self._create_issue_from_dict(current_issue)
            if issue:
                issue.relevance_score = self._calculate_relevance(issue, keywords)
                issues.append(issue)

        return issues

    def _create_issue_from_dict(self, issue_dict: Dict[str, str]) -> Optional[IssueItem]:
        """딕셔너리로부터 IssueItem 생성"""
        if not issue_dict.get('title'):
            return None

        return IssueItem(
            title=issue_dict['title'],
            summary=issue_dict.get('요약', issue_dict['title']),
            source=issue_dict.get('출처', 'Unknown'),
            published_date=issue_dict.get('발행일'),
            category=issue_dict.get('카테고리', 'general'),
            relevance_score=0.5
        )

    def _parse_issue_section(self, section: str) -> Optional[IssueItem]:
        """개별 이슈 블록을 파싱하여 IssueItem 객체를 생성합니다."""
        try:
            title_match = re.search(r'##\s*\*\*(.*?)\*\*', section)
            if not title_match: return None
            title = title_match.group(1).strip()
            summary = self._extract_field(section, '요약')
            source = self._extract_field(section, '출처') or 'Unknown'
            date_str = self._extract_field(section, '발행일')
            category = self._extract_field(section, '카테고리') or 'general'
            url_match = re.search(r'https?://[^\s)]+', source)
            url = url_match.group(0) if url_match else None
            return IssueItem(title=title, summary=summary or title, source=source, published_date=date_str, relevance_score=0.5, category=category, url=url)
        except Exception as e:
            logger.error(f"개별 이슈 섹션 파싱 실패: {e}")
            return None

    def _extract_field(self, text: str, field_name: str) -> Optional[str]:
        """정규식을 사용해 특정 필드의 내용을 추출합니다."""
        pattern = rf'\*\*{field_name}\*\*:\s*(.*?)(?=\n\*\*|\Z)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _calculate_relevance(self, issue: IssueItem, keywords: List[str]) -> float:
        """이슈의 제목과 요약에 키워드가 얼마나 포함되었는지로 관련도를 계산합니다."""
        text = f"{issue.title} {issue.summary}".lower()
        score = sum(1.0 - (i * 0.1) for i, keyword in enumerate(keywords) if keyword.lower() in text)
        return round(min(score / len(keywords), 1.0) if keywords else 0.0, 2)


# --- 메인 실행 로직 ---
async def main():
    """스크립트의 메인 실행 함수입니다."""
    if not os.getenv('PERPLEXITY_API_KEY') or not os.getenv('ANTHROPIC_API_KEY'):
        print("⛔️ 오류: .env 파일에 'PERPLEXITY_API_KEY'와 'ANTHROPIC_API_KEY'를 설정해야 합니다.")
        return

    searcher = ClaudeIssueSearcher()
    # 검색할 주제어입니다.
    topic = "iOS"
    # 변경된 부분: 오늘로부터 며칠 전까지의 이슈를 검색할지 숫자로 지정합니다. (예: 30은 최근 30일)
    days_to_search = 30

    result = await searcher.search(topic, days_ago=days_to_search, analyze=True)

    search_result = result["search_result"]
    analysis = result["analysis"]
    output_filename = f"{topic.replace(' ', '_')}_이슈_분석_결과.txt"
    original_stdout = sys.stdout

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            sys.stdout = f

            print(f"주제: '{topic}'에 대한 이슈 분석 보고서\n" + "="*50)
            # 추가된 부분: 검색 기간을 명시적으로 출력
            print(f"✔️ 검색 기간: {search_result.search_start_date} ~ {search_result.search_end_date} ({days_to_search}일간)")
            print(f"✔️ 사용된 키워드: {', '.join(search_result.keywords)}")
            print(f"✔️ 1차 검색된 이슈: {search_result.initial_found}개")
            print(f"✔️ 최종 검증된 이슈: {search_result.verified_found}개 (신뢰도 향상)")
            print(f"✔️ 총 소요 시간: {search_result.search_time:.2f}초\n")

            if not search_result.issues:
                print("최종적으로 검증된 이슈가 없습니다.")
            else:
                print(f"--- 상위 {min(5, len(search_result.issues))}개 검증된 이슈 상세 정보 ---\n")
                for i, issue in enumerate(search_result.issues[:5], 1):
                    print(f"{i}. {issue.title}")
                    print(f"   - 출처: {issue.source}")
                    print(f"   - 날짜: {issue.published_date}")
                    print(f"   - 관련도: {issue.relevance_score:.1%}")
                    print(f"   - 검증: {issue.verification.status} ({issue.verification.reasoning})")
                    print(f"   - 요약: {issue.summary}\n")

            if analysis and analysis.get('full_analysis'):
                print("\n" + "--- Claude 심층 분석 결과 ---\n" + "="*50)
                print(analysis['full_analysis'])

        sys.stdout = original_stdout
        print(f"\n🎉 성공! 분석 결과가 '{output_filename}' 파일에 안전하게 저장되었습니다.")

    except Exception as e:
        sys.stdout = original_stdout
        logger.error(f"결과 파일 저장 중 오류 발생: {e}")
        print(f"⛔️ 오류: 분석 결과를 파일에 저장하지 못했습니다. ({e})")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="DEBUG")

    print("🚀 지능형 이슈 검색 시스템을 시작합니다...")
    print("⚠️ Perplexity API는 짧은 시간에 많은 요청을 보내면 사용량 제한에 걸릴 수 있으니 주의하세요.")

    asyncio.run(main())