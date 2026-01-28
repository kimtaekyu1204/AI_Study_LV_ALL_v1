"""
웹 통신의 기초 개념 정리

주제: HTTP 요청-응답 메커니즘, 클라이언트-서버 모델

작성일: 2025-01-28
출처: Web 기초 (AI2501).pptx
"""

# ============================================================================
# 1. 웹 통신의 기본 개념
# ============================================================================

"""
웹이란:
- 전세계 컴퓨터들이 인터넷을 통해 연결된 정보 시스템
- 클라이언트-서버 아키텍처로 구성됨
- HTTP 프로토콜을 기반으로 통신함

웹 통신의 핵심:
- 요청(Request)과 응답(Response)의 반복
- 클라이언트가 요청하면 서버가 응답한다.
- 이러한 상호작용을 통해 데이터를 주고받는다.
"""

# ============================================================================
# 2. 클라이언트-서버 모델
# ============================================================================

"""
클라이언트-서버 아키텍처:

클라이언트 (Client):
- 정의: 사용자가 직접 사용하는 프로그램 (웹 브라우저 등)
- 역할: 서버에 요청을 보낸다.
- 예시:
  * Google Chrome, Firefox, Safari 등의 웹 브라우저
  * 모바일 앱
  * 데스크톱 애플리케이션

서버 (Server):
- 정의: 클라이언트의 요청을 처리하고 응답을 제공하는 컴퓨터
- 역할: 클라이언트의 요청을 받아 처리 후 응답을 보낸다.
- 예시:
  * 웹 서버 (Apache, Nginx)
  * 애플리케이션 서버
  * 데이터베이스 서버

통신 과정:
1. 클라이언트가 특정 URL 입력 또는 버튼 클릭
2. 요청(Request) 생성 및 전송
3. 서버가 요청을 수신하고 처리
4. 응답(Response) 생성 및 전송
5. 클라이언트가 응답 수신 및 화면 표시
"""

# ============================================================================
# 3. HTTP 요청 (Request)의 구조
# ============================================================================

"""
HTTP 요청의 구성 요소:

1. 요청 메서드 (Request Method)
   정의: 서버에 무엇을 하려고 하는지 알려주는 명령

   주요 메서드:
   - GET: 서버에서 데이터를 가져온다 (조회)
     * 용도: 정보 검색, 페이지 로드
     * 특징: URL에 파라미터를 포함할 수 있음
     * 예: https://www.google.com/search?q=python

   - POST: 서버에 데이터를 보낸다 (생성/수정)
     * 용도: 폼 제출, 새 데이터 생성
     * 특징: 본문(body)에 데이터를 포함
     * 예: 회원가입 양식 전송

   - PUT: 기존 데이터를 전체 수정한다
   - DELETE: 서버의 데이터를 삭제한다
   - PATCH: 데이터의 일부를 수정한다

2. 요청 URL (Request URL)
   정의: 요청을 보낼 서버의 주소

   URL의 구조:
   https://www.example.com:8080/api/user?id=123#section
   ├─ https: 프로토콜 (HTTPS는 보안 강화 버전)
   ├─ www.example.com: 도메인
   ├─ :8080: 포트 번호 (생략 시 80 또는 443)
   ├─ /api/user: 경로 (서버의 특정 리소스)
   ├─ ?id=123: 쿼리 문자열 (파라미터)
   └─ #section: 프래그먼트

3. HTTP 버전 (HTTP Version)
   - HTTP/1.1: 널리 사용되는 표준
   - HTTP/2: 성능 개선 버전
   - HTTP/3: 최신 버전, 더욱 빠른 속도

4. 요청 헤더 (Request Headers)
   정의: 요청에 대한 추가 정보를 담고 있다

   예시:
   - User-Agent: 클라이언트의 정보 (브라우저, OS)
   - Content-Type: 데이터의 형식 (application/json 등)
   - Authorization: 인증 정보 (토큰, API 키)
   - Cookie: 이전에 저장된 쿠키 정보
   - Accept: 클라이언트가 받을 수 있는 데이터 형식

5. 요청 본문 (Request Body)
   정의: 서버로 보낼 데이터 (주로 POST/PUT 요청에서 사용)

   예시:
   {
       "username": "john_doe",
       "email": "john@example.com",
       "password": "secure_password"
   }
"""

# ============================================================================
# 4. HTTP 응답 (Response)의 구조
# ============================================================================

"""
HTTP 응답의 구성 요소:

1. HTTP 버전 (HTTP Version)
   - 요청과 동일한 버전으로 응답한다
   - 예: HTTP/1.1, HTTP/2

2. 상태 코드 (Response Status Code)
   정의: 요청의 처리 결과를 나타내는 3자리 숫자

   상태 코드의 분류:
   - 1xx (정보): 요청을 받았으며 진행 중
   - 2xx (성공): 요청이 성공적으로 처리됨
     * 200 OK: 요청 성공
     * 201 Created: 새 리소스 생성 성공
     * 204 No Content: 요청 성공, 응답 본문 없음

   - 3xx (리다이렉션): 추가 조치가 필요
     * 301 Moved Permanently: 영구적으로 다른 URL로 이동
     * 302 Found: 임시로 다른 URL로 이동
     * 304 Not Modified: 캐시된 데이터 사용

   - 4xx (클라이언트 오류): 클라이언트의 잘못된 요청
     * 400 Bad Request: 잘못된 요청 형식
     * 401 Unauthorized: 인증 필요
     * 403 Forbidden: 접근 권한 없음
     * 404 Not Found: 요청한 리소스가 없음
     * 429 Too Many Requests: 요청 횟수 초과

   - 5xx (서버 오류): 서버의 문제
     * 500 Internal Server Error: 서버 내부 오류
     * 502 Bad Gateway: 게이트웨이 오류
     * 503 Service Unavailable: 서버 점검 중

3. 상태 텍스트 (Response Status Text)
   정의: 상태 코드를 설명하는 텍스트

   예시:
   - 200 OK
   - 404 Not Found
   - 500 Internal Server Error

4. 응답 헤더 (Response Headers)
   정의: 응답에 대한 추가 정보

   예시:
   - Content-Type: 응답 데이터의 형식 (text/html, application/json)
   - Content-Length: 응답 본문의 크기
   - Set-Cookie: 클라이언트에 저장할 쿠키
   - Cache-Control: 캐시 관련 설정
   - Server: 서버의 정보

5. 응답 본문 (Response Body)
   정의: 서버가 클라이언트에게 보내는 실제 데이터

   예시 (HTML):
   <!DOCTYPE html>
   <html>
   <head><title>Welcome</title></head>
   <body>Hello, World!</body>
   </html>

   예시 (JSON):
   {
       "id": 1,
       "name": "John Doe",
       "email": "john@example.com"
   }
"""

# ============================================================================
# 5. 요청-응답 사이클 (Request-Response Cycle)
# ============================================================================

"""
웹 통신의 전체 흐름:

클라이언트                           서버
   |                                 |
   |--- Request 생성 및 전송 ------->  |
   |                                 |
   |  요청 메서드: GET               |
   |  요청 URL: /api/users           |
   |  HTTP 버전: HTTP/1.1            |
   |  헤더 정보 포함                 |
   |                                 |
   |                            처리 시작
   |                                 |
   |                            데이터베이스 조회
   |                                 |
   |                            응답 준비
   |                                 |
   |  <----- Response 전송 --------- |
   |                                 |
   |  상태 코드: 200 OK              |
   |  Content-Type: application/json |
   |  응답 본문: JSON 데이터         |
   |                                 |
   |  화면 렌더링 및 표시            |
   |                                 |
"""

# ============================================================================
# 6. 예제 코드 (Python requests 라이브러리)
# ============================================================================

# Python에서 웹 통신을 하는 예제 (requests 라이브러리 사용)

def example_http_request():
    """
    HTTP 요청 예제
    (실제 실행을 위해서는 'pip install requests' 필요)
    """
    import requests

    # 1. GET 요청 예제
    # 정의: 서버에서 데이터를 가져온다
    # 용도: 웹 페이지, API에서 정보 조회

    print("=== GET 요청 예제 ===")
    # 요청 URL, 메서드, 헤더 등이 자동으로 조합됨
    response = requests.get('https://api.github.com/users/github')

    # 응답 정보
    print(f"상태 코드: {response.status_code}")  # 200 OK
    print(f"응답 헤더: {response.headers}")
    print(f"응답 본문: {response.json()}")  # JSON으로 파싱

    # 2. POST 요청 예제
    # 정의: 서버에 새로운 데이터를 전송한다
    # 용도: 사용자 정보 생성, 폼 제출

    print("\n=== POST 요청 예제 ===")
    # 전송할 데이터
    data = {
        "title": "새로운 게시물",
        "body": "이것은 테스트 게시물입니다.",
        "userId": 1
    }

    # POST 요청 (데이터 포함)
    response = requests.post(
        'https://jsonplaceholder.typicode.com/posts',
        json=data,  # 요청 본문으로 JSON 데이터 전송
        headers={
            'Content-Type': 'application/json'  # 데이터 형식 명시
        }
    )

    print(f"상태 코드: {response.status_code}")  # 201 Created
    print(f"생성된 리소스: {response.json()}")

    # 3. 파라미터를 포함한 GET 요청
    print("\n=== 파라미터 포함 GET 요청 ===")
    params = {
        'page': 1,
        'limit': 10
    }

    response = requests.get(
        'https://jsonplaceholder.typicode.com/posts',
        params=params  # URL에 자동으로 추가됨
    )

    print(f"상태 코드: {response.status_code}")
    print(f"조회된 게시물 수: {len(response.json())}")


# ============================================================================
# 7. 실제 웹 통신의 과정 (상세)
# ============================================================================

"""
브라우저 주소창에 'https://www.google.com'을 입력할 때:

1단계: URL 파싱
   - 프로토콜: https
   - 도메인: www.google.com
   - 포트: 443 (HTTPS 기본 포트)

2단계: DNS 조회
   - www.google.com을 IP 주소로 변환
   - 예: 142.250.185.46

3단계: 요청 생성
   - 메서드: GET
   - URL: https://www.google.com/
   - 헤더 정보 추가 (User-Agent, Accept 등)

4단계: 요청 전송
   - TCP/IP를 통해 네트워크로 전송

5단계: 서버 처리
   - 구글 서버에서 요청 수신
   - 요청 분석 및 처리
   - 검색 페이지 HTML 생성

6단계: 응답 전송
   - 상태 코드: 200 OK
   - Content-Type: text/html
   - HTML 본문 전송

7단계: 클라이언트에서 수신
   - HTML 파싱
   - CSS, JavaScript 로드
   - 이미지 등의 리소스 로드

8단계: 렌더링
   - 구글 검색 페이지 화면에 표시
"""

# ============================================================================
# 8. 요점 정리
# ============================================================================

"""
웹 통신의 핵심:

1. 클라이언트-서버 모델
   - 클라이언트가 요청하고 서버가 응답한다.

2. HTTP 요청의 구성
   - 메서드: GET, POST, PUT, DELETE 등
   - URL: 리소스의 위치
   - 헤더: 추가 정보
   - 본문: 데이터 (선택사항)

3. HTTP 응답의 구성
   - 상태 코드: 요청 처리 결과
   - 헤더: 응답에 대한 정보
   - 본문: 실제 데이터

4. 상태 코드의 의미
   - 2xx: 성공
   - 3xx: 리다이렉션
   - 4xx: 클라이언트 오류
   - 5xx: 서버 오류

5. 웹 통신은 요청-응답의 반복으로 이루어진다.
"""

if __name__ == "__main__":
    print("=== 웹 기초 개념 ===\n")
    print("클라이언트 ---요청---> 서버")
    print("클라이언트 <---응답--- 서버\n")
    print("HTTP 요청 메서드:")
    print("  - GET: 데이터 조회")
    print("  - POST: 데이터 생성")
    print("  - PUT: 데이터 전체 수정")
    print("  - DELETE: 데이터 삭제\n")
    print("HTTP 상태 코드:")
    print("  - 2xx: 성공 (200 OK)")
    print("  - 4xx: 클라이언트 오류 (404 Not Found)")
    print("  - 5xx: 서버 오류 (500 Internal Server Error)")
