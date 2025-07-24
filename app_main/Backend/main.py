"""
Claude Opus 4를 사용한 간단한 이슈 검색 시스템 (하이브리드 방식)
"""

import asyncio
import re
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import httpx
from loguru import logger
import anthropic
import os
from dotenv import load_dotenv
import sys # sys 모듈 임포트

# 환경 변수 로드
load_dotenv()


# ============= 데이터 모델 =============
@dataclass
class IssueItem:
    """검색된 이슈 아이템"""
    title: str
    summary: str
    source: str
    published_date: Optional[str]
    relevance_score: float
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


# ============= 키워드 생성 (하이브리드 방식에서는 직접 사용되지 않음) =============
class ClaudeKeywordGenerator:
    """Claude Opus 4를 사용한 키워드 생성기 (하이브리드 방식에서는 직접 사용되지 않음)"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key) if self.api_key else None
        self.model = "claude-opus-4-20250514"  # Claude Opus 4 모델

    async def generate_keywords(self, topic: str, max_keywords: int = 10) -> List[str]:
        """주제에 대한 키워드 생성"""
        if not self.client:
            logger.warning("Anthropic API 키가 없어 기본 키워드 사용")
            return self._generate_basic_keywords(topic)

        try:
            prompt = f"""
주제: "{topic}"

이 주제와 관련된 검색 키워드를 {max_keywords}개 생성해주세요.
- 핵심 키워드 3-4개
- 관련 용어 3-4개  
- 최신 트렌드 키워드 2-3개

한 줄에 하나씩, 키워드만 작성해주세요.
"""
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.7,
                system="당신은 키워드 생성 전문가입니다. 주제에 대한 정확하고 관련성 높은 키워드를 생성합니다.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            keywords = message.content[0].text.strip().split('\n')
            keywords = [kw.strip() for kw in keywords if kw.strip()]

            # 원본 토픽도 포함
            if topic not in keywords:
                keywords.insert(0, topic)

            return keywords[:max_keywords]

        except Exception as e:
            logger.error(f"키워드 생성 실패: {e}")
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
        """이슈 검색"""
        # 하이브리드 방식에서는 확장된 키워드를 사용하여 검색 프롬프트 생성
        prompt = f"""
'{", ".join(keywords)}' 키워드와 관련하여 '{time_period}' 동안 발행된 주요 이슈를 찾아주세요.

각 이슈마다 다음 형식으로 작성해주세요:

## **[이슈 제목]**
**요약**: [간단한 요약]
**출처**: [웹사이트명 또는 URL]
**발행일**: [YYYY-MM-DD 형식]
**카테고리**: [뉴스/기술/비즈니스 등]

최대 10개의 이슈를 찾아주세요.
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 정확한 정보 분석 전문가입니다. 실제 출처와 날짜를 포함해 주세요."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 4000,
            "temperature": 0.2
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()


# ============= Claude를 사용한 향상된 분석 =============
class ClaudeAnalyzer:
    """Claude를 사용한 이슈 분석 및 요약"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key) if self.api_key else None
        self.model = "claude-opus-4-20250514"

    async def analyze_issues(self, issues: List[IssueItem], topic: str) -> Dict[str, Any]:
        """수집된 이슈들을 분석하고 인사이트 도출"""
        if not self.client or not issues:
            return {"summary": "분석 불가", "insights": []}

        # 이슈 내용 준비
        issues_text = "\n\n".join([
            f"제목: {issue.title}\n요약: {issue.summary}\n출처: {issue.source}\n날짜: {issue.published_date}"
            for issue in issues[:5]  # 상위 5개만 분석
        ])

        prompt = f"""
주제: "{topic}"

다음은 관련 이슈들입니다:
{issues_text}

위 이슈들을 분석하여 다음을 제공해주세요:
1. 전체적인 트렌드 요약 (2-3문장)
2. 주요 인사이트 3개
3. 향후 예상 동향 (1-2문장)

간결하고 명확하게 작성해주세요.
"""

        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.5,
                system="당신은 기술 트렌드 분석 전문가입니다. 이슈들을 종합적으로 분석하여 인사이트를 도출합니다.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            analysis_text = message.content[0].text

            # 간단한 파싱
            return {
                "summary": analysis_text.split("\n")[0] if analysis_text else "분석 결과 없음",
                "full_analysis": analysis_text,
                "analyzed_count": len(issues[:5])
            }

        except Exception as e:
            logger.error(f"Claude 분석 실패: {e}")
            return {"summary": "분석 실패", "insights": []}


# ============= 통합 이슈 검색기 =============
class ClaudeIssueSearcher:
    """Claude Opus 4를 활용한 이슈 검색기"""

    def __init__(self, anthropic_key: Optional[str] = None, perplexity_key: Optional[str] = None):
        self.keyword_generator = ClaudeKeywordGenerator(anthropic_key) # 여전히 존재하지만 사용되지 않음
        self.perplexity_client = SimplePerplexityClient(perplexity_key)
        self.analyzer = ClaudeAnalyzer(anthropic_key)

    async def search(self, topic: str, time_period: str = "최근 1주일", analyze: bool = True) -> Dict[str, Any]:
        """주제에 대한 이슈 검색 및 분석"""
        start_time = time.time()

        try:
            # 1. 하이브리드 방식으로 키워드 확장 (API 호출 없이)
            logger.info(f"하이브리드 방식으로 키워드 확장 중: {topic}")
            expanded_keywords = [
                topic,
                f"{topic} 최신",
                f"{topic} 동향",
                f"{topic} 이슈"
            ]
            keywords = expanded_keywords # 이 키워드들을 검색 및 결과 객체에 사용
            logger.info(f"확장된 키워드: {keywords}")

            # 2. Perplexity로 이슈 검색
            logger.info("이슈 검색 중...")
            api_response = await self.perplexity_client.search_issues(keywords, time_period)

            # 3. 응답 파싱
            issues = self._parse_response(api_response, keywords)

            # 4. Claude로 분석 (옵션)
            analysis = None
            if analyze and issues:
                logger.info("Claude로 이슈 분석 중...")
                analysis = await self.analyzer.analyze_issues(issues, topic)

            # 5. 결과 반환
            search_time = time.time() - start_time
            logger.info(f"검색 완료: {len(issues)}개 이슈 발견 ({search_time:.2f}초)")

            result = SearchResult(
                topic=topic,
                keywords=keywords, # 확장된 키워드를 결과에 포함
                issues=issues,
                total_found=len(issues),
                search_time=search_time
            )

            return {
                "search_result": result,
                "analysis": analysis
            }

        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return {
                "search_result": SearchResult(
                    topic=topic,
                    keywords=[topic],
                    issues=[],
                    total_found=0,
                    search_time=time.time() - start_time
                ),
                "analysis": None
            }

    def _parse_response(self, api_response: Dict[str, Any], keywords: List[str]) -> List[IssueItem]:
        """API 응답 파싱"""
        try:
            content = api_response['choices'][0]['message']['content']

            # 이슈 섹션 분리
            issue_blocks = re.finditer(r'(?s)(##\s*\*\*.*?(?=\n##\s*\*\*|\Z))', content)
            issues = []

            for match in issue_blocks:
                section = match.group(1).strip()
                issue = self._parse_issue_section(section)
                if issue:
                    # 관련성 점수 계산
                    issue.relevance_score = self._calculate_relevance(issue, keywords)
                    issues.append(issue)

            # 관련성 점수로 정렬
            issues.sort(key=lambda x: x.relevance_score, reverse=True)
            return issues[:10]  # 상위 10개만 반환

        except Exception as e:
            logger.error(f"응답 파싱 실패: {e}")
            return []

    def _parse_issue_section(self, section: str) -> Optional[IssueItem]:
        """이슈 섹션 파싱"""
        try:
            # 제목 추출
            title_match = re.search(r'##\s*\*\*(.*?)\*\*', section)
            if not title_match:
                return None
            title = title_match.group(1).strip()

            # 필드 추출
            summary = self._extract_field(section, '요약')
            source = self._extract_field(section, '출처') or 'Unknown'
            date_str = self._extract_field(section, '발행일')
            category = self._extract_field(section, '카테고리') or 'general'

            # URL 추출 (있으면)
            url_match = re.search(r'https?://[^\s]+', source)
            url = url_match.group(0) if url_match else None

            return IssueItem(
                title=title,
                summary=summary or title,
                source=source,
                published_date=date_str,
                relevance_score=0.5,  # 초기값
                category=category,
                url=url
            )

        except Exception as e:
            logger.error(f"이슈 파싱 실패: {e}")
            return None

    def _extract_field(self, text: str, field_name: str) -> Optional[str]:
        """필드 값 추출"""
        pattern = rf'\*\*{field_name}\*\*:\s*(.*?)(?=\*\*|$)'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _calculate_relevance(self, issue: IssueItem, keywords: List[str]) -> float:
        """관련성 점수 계산"""
        text = f"{issue.title} {issue.summary}".lower()
        score = 0.0

        # 키워드 매칭
        for i, keyword in enumerate(keywords):
            if keyword.lower() in text:
                # 앞쪽 키워드일수록 높은 가중치
                weight = 1.0 - (i * 0.1)
                score += max(weight, 0.1)

        # 정규화 (0.0 ~ 1.0)
        normalized_score = min(score / len(keywords), 1.0)
        return round(normalized_score, 2)


# ============= 사용 예제 =============
async def main():
    """사용 예제"""
    # 검색기 초기화 (Claude Opus 4 사용)
    searcher = ClaudeIssueSearcher()

    # 이슈 검색 및 분석
    topic = "iOS"
    result = await searcher.search(topic, "최근 1주일", analyze=True)

    search_result = result["search_result"]
    analysis = result["analysis"]

    # 파일로 저장할 경로 및 파일명 설정
    output_filename = f"{topic}_search_results.txt"

    # 기존 stdout을 저장
    original_stdout = sys.stdout

    try:
        # 파일을 쓰기 모드로 열고 stdout을 파일로 리디렉션
        with open(output_filename, "w", encoding="utf-8") as f:
            sys.stdout = f # stdout을 파일 객체로 변경

            # 결과 출력
            print(f"\n=== '{topic}' 검색 결과 ===")
            print(f"키워드: {', '.join(search_result.keywords)}")
            print(f"발견된 이슈: {search_result.total_found}개")
            print(f"검색 시간: {search_result.search_time:.2f}초\n")

            # 이슈 출력
            for i, issue in enumerate(search_result.issues[:5], 1):
                print(f"{i}. {issue.title}")
                print(f"   출처: {issue.source}")
                print(f"   날짜: {issue.published_date}")
                print(f"   관련도: {issue.relevance_score:.1%}")
                print(f"   요약: {issue.summary}")
                print()

            # Claude 분석 결과 출력
            if analysis:
                print("\n=== Claude Opus 4 분석 결과 ===")
                print(f"분석 요약: {analysis.get('summary', 'N/A')}")
                if 'full_analysis' in analysis:
                    print(f"\n상세 분석:\n{analysis['full_analysis']}")

        # 파일 저장이 완료되었음을 사용자에게 알림 (콘솔에 출력)
        sys.stdout = original_stdout # stdout을 다시 콘솔로 복원
        print(f"\n검색 결과가 '{output_filename}' 파일에 저장되었습니다.")

    except Exception as e:
        sys.stdout = original_stdout # 에러 발생 시에도 stdout 복원
        logger.error(f"파일 저장 중 오류 발생: {e}")
        print(f"오류 발생: 검색 결과를 파일에 저장할 수 없습니다. {e}")


# 실행
if __name__ == "__main__":
    asyncio.run(main())
