import requests
import json
import time
import os

# 디버그 모드 활성화 (통계 정보 포함)
os.environ['DEBUG_MODE'] = 'true'


def test_search_api_improved(topic, timeout=120):
    """
    개선된 API 테스트
    """
    url = "http://127.0.0.1:8000/api/search"
    headers = {"Content-Type": "application/json"}

    print(f"\n{'=' * 60}")
    print(f"검색 주제: {topic}")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        response = requests.post(
            url,
            headers=headers,
            json={
                "topic": topic,
                "time_period": "최근 1주일",
                "analyze": True
            },
            timeout=timeout
        )

        elapsed_time = time.time() - start_time

        print(f"응답 시간: {elapsed_time:.2f}초")
        print(f"상태 코드: {response.status_code}")

        if response.status_code == 200:
            result = response.json()

            # 결과 분석
            print(f"\n[결과 요약]")
            print(f"제목: {result.get('제목', 'N/A')}")
            print(f"태그: {', '.join(result.get('태그', []))}")

            report = result.get('보고서', {})

            # 출처 링크 분석 (수정된 부분)
            links = report.get('출처 링크', [])
            # 딕셔너리 리스트에서 URL만 추출하여 중복 제거
            if links and isinstance(links[0], dict):
                unique_urls = list(set(link.get('url', '') for link in links if link.get('url')))
                print(f"\n[수집된 이슈]")
                print(f"전체 링크: {len(links)}개")
                print(f"고유 URL: {len(unique_urls)}개")

                # 링크 샘플 출력
                print("\n[링크 샘플 (최대 3개)]")
                for i, link in enumerate(links[:3]):
                    print(f"  {i + 1}. {link.get('title', 'N/A')}")
                    print(f"     URL: {link.get('url', 'N/A')}")
            else:
                print(f"\n[수집된 이슈]")
                print(f"링크 형식 오류 또는 링크 없음")

            # 요약 내용 분석
            content = report.get('정리된 내용', '')
            summaries = []
            if content and content != "유효한 요약 내용이 없습니다.":
                # ### 으로 구분된 섹션 파싱
                sections = content.split('###')
                summaries = [s.strip() for s in sections[1:] if s.strip()]  # 첫 번째는 빈 문자열
                print(f"유효한 요약: {len(summaries)}개")
            else:
                print("유효한 요약: 0개")

            # AI 분석 확인
            ai_report = report.get('AI가 제공하는 리포트', '')
            if ai_report and ai_report not in ["분석 내용이 없습니다.", "종합 분석을 생성하지 못했습니다."]:
                print(f"AI 분석: 있음 ({len(ai_report)}자)")
                # AI 분석 내용 일부 출력
                print("\n[AI 분석 요약]")
                print(ai_report)
            else:
                print("AI 분석: 없음")

            # 통계 정보 (DEBUG_MODE가 true일 때만)
            stats = report.get('통계')
            if stats:
                print(f"\n[통계 정보]")
                for key, value in stats.items():
                    print(f"- {key}: {value}")

            # 샘플 출력
            if summaries and len(summaries) > 0:
                print(f"\n[첫 번째 요약 샘플]")
                first_summary = summaries[0]
                # 제목과 내용 분리
                lines = first_summary.split('\n', 1)
                if len(lines) > 0:
                    print(f"제목: {lines[0]}")
                if len(lines) > 1:
                    content_preview = lines[1].strip()
                    print(f"내용: {content_preview}..." if len(content_preview) > 200 else f"내용: {content_preview}")

            # 전체 응답 구조 확인 (디버깅용)
            if os.getenv('FULL_DEBUG', '').lower() == 'true':
                print("\n[전체 응답 구조]")
                print(json.dumps(result, indent=2, ensure_ascii=False) + "...")

        else:
            print(f"\n오류 응답:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))

    except requests.exceptions.Timeout:
        print(f"요청 시간 초과 ({timeout}초)")
    except requests.exceptions.ConnectionError:
        print("서버에 연결할 수 없습니다.")
    except Exception as e:
        print(f"예상치 못한 오류: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    return elapsed_time if 'elapsed_time' in locals() else None


# 서버 상태 확인
def check_server_health():
    """서버 상태 확인"""
    try:
        response = requests.get("http://127.0.0.1:8000/")
        if response.status_code == 200:
            print("서버 상태: 정상")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print(f"서버 상태: 비정상 (코드: {response.status_code})")
    except Exception as e:
        print(f"서버에 연결할 수 없습니다: {e}")


# 연속 테스트
def run_multiple_tests():
    """여러 주제로 연속 테스트"""
    topics = ["머신러닝", "딥러닝", "자연어처리"]

    print("\n" + "=" * 60)
    print("연속 테스트 시작")
    print("=" * 60)

    results = []
    for i, topic in enumerate(topics):
        if i > 0:
            print("\n5초 대기 중...")
            time.sleep(5)

        elapsed = test_search_api_improved(topic)
        if elapsed:
            results.append((topic, elapsed))

    # 결과 요약
    if results:
        print("\n" + "=" * 60)
        print("테스트 결과 요약")
        print("=" * 60)
        for topic, elapsed in results:
            print(f"- {topic}: {elapsed:.2f}초")
        avg_time = sum(t for _, t in results) / len(results)
        print(f"\n평균 응답 시간: {avg_time:.2f}초")


# 테스트 실행
if __name__ == "__main__":
    # 서버 상태 확인
    print("서버 상태 확인 중...")
    check_server_health()

    # 단일 테스트
    test_search_api_improved("인공지능")

    # 연속 테스트 (주석 해제하여 사용)
    run_multiple_tests()