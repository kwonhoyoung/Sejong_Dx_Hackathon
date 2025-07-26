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

            # 출처 링크 분석
            links = report.get('출처 링크', [])
            unique_links = list(set(links))
            print(f"\n[수집된 이슈]")
            print(f"전체 링크: {len(links)}개")
            print(f"고유 링크: {len(unique_links)}개")

            # 요약 내용 분석
            content = report.get('정리된 내용', '')
            if content and content != "유효한 요약 내용이 없습니다.":
                summaries = content.split('###')[1:]  # 첫 번째는 빈 문자열
                print(f"유효한 요약: {len(summaries)}개")
            else:
                print("유효한 요약: 0개")

            # AI 분석 확인
            ai_report = report.get('AI가 제공하는 리포트', '')
            if ai_report and ai_report != "분석 내용이 없습니다.":
                print(f"AI 분석: 있음 ({len(ai_report)}자)")
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
                first_summary = summaries[0].strip()
                print(first_summary[:200] + "..." if len(first_summary) > 200 else first_summary)

        else:
            print(f"\n오류 응답:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))

    except requests.exceptions.Timeout:
        print(f"요청 시간 초과 ({timeout}초)")
    except requests.exceptions.ConnectionError:
        print("서버에 연결할 수 없습니다.")
    except Exception as e:
        print(f"예상치 못한 오류: {type(e).__name__}: {e}")

    return elapsed_time if 'elapsed_time' in locals() else None


# 테스트 실행
if __name__ == "__main__":
    # 단일 테스트
    test_search_api_improved("인공지능")

    # 연속 테스트 (API 속도 제한 고려)
    print("\n\n" + "=" * 60)
    print("연속 테스트 시작")
    print("=" * 60)

    topics = ["머신러닝", "딥러닝", "자연어처리"]
    for i, topic in enumerate(topics):
        if i > 0:
            print("\n5초 대기 중...")
            time.sleep(5)

        test_search_api_improved(topic)


# 서버 상태 확인
def check_server_health():
    """서버 상태 확인"""
    try:
        response = requests.get("http://127.0.0.1:8000/health")
        if response.status_code == 200:
            print("서버 상태: 정상")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"서버 상태: 비정상 (코드: {response.status_code})")
    except:
        print("서버에 연결할 수 없습니다.")

# 사용법:
# check_server_health()