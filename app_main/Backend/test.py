import requests
import json

# API 엔드포인트 URL
url = "http://127.0.0.1:8000/api/search"

# 요청 헤더
headers = {
    "Content-Type": "application/json"
}

# 요청 데이터
data = {
    "topic": "인공지능"
}

try:
    # POST 요청 보내기
    response = requests.post(url, headers=headers, json=data)

    # 응답 상태 코드 확인
    print(f"상태 코드: {response.status_code}")

    # 응답 헤더 출력 (선택사항)
    print(f"응답 헤더: {response.headers}")

    # 응답 본문 출력
    if response.status_code == 200:
        # JSON 응답인 경우
        try:
            result = response.json()
            print("응답 데이터 (JSON):")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            # JSON이 아닌 경우 텍스트로 출력
            print("응답 데이터 (텍스트):")
            print(response.text)
    else:
        print(f"오류 발생: {response.text}")

except requests.exceptions.ConnectionError:
    print("서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
except requests.exceptions.Timeout:
    print("요청 시간이 초과되었습니다.")
except Exception as e:
    print(f"예상치 못한 오류 발생: {e}")


# 추가 테스트를 위한 함수 버전
def search_api(topic):
    """
    API에 검색 요청을 보내는 함수

    Args:
        topic (str): 검색할 주제

    Returns:
        dict or str: 응답 데이터
    """
    try:
        response = requests.post(url, headers=headers, json={"topic": topic})
        response.raise_for_status()  # HTTP 오류가 있으면 예외 발생

        try:
            return response.json()
        except json.JSONDecodeError:
            return response.text

    except requests.exceptions.RequestException as e:
        print(f"API 요청 중 오류 발생: {e}")
        return None


# 함수 사용 예시
print("\n--- 함수 테스트 ---")
result = search_api("인공지능")
if result:
    print("검색 결과:", result)

"""
# 다른 주제로 테스트
print("\n--- 다른 주제 테스트 ---")
test_topics = ["머신러닝", "딥러닝", "자연어처리"]
for topic in test_topics:
    print(f"\n'{topic}' 검색 중...")
    result = search_api(topic)
    if result:
        print(f"결과: {result}")
"""