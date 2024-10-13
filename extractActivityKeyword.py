import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as VectorStorePinecone
from pinecone import Pinecone

from outputParser import activityKeyword_parser

load_dotenv()

pc = Pinecone(
    api_key=os.environ['PINECONE_API_KEY']
)

index = pc.Index('keywords')
embeddings = OpenAIEmbeddings()
vectorstore = VectorStorePinecone.from_existing_index('keywords', embeddings)
llm = ChatOpenAI(
    model=os.environ['GOAL_MODEL'],
    temperature=os.environ['GOAL_TEMP']
)

format_instructions = activityKeyword_parser.get_format_instructions()
template = """
질문: 다음은 플레너의 이름과 플레너의 설명입니다. 이름과 설명을 읽고 해당 플레너의 활동 키워드를 제공된 단어 안에서 추출하세요.
활동 키워드란 플래너에서 투두로 해야 할 것들의 키워드입니다. 

플레너 이름: {name}
플레너 설명: {desc}

제공 단어:
"읽기", "쓰기", "코딩하기", "연습하기", "운동하기", "공부하기", "정리하기", "노트하기", "분석하기",
"테스트하기", "강의듣기", "실험하기", "프로젝트하기", "설계하기", "디자인하기", "회의하기", "발표하기",
"작성하기", "계획하기", "조사하기", "연습문제 풀기", "피드백 받기", "코드 리뷰하기", "발표 준비하기",
"자료 정리하기", "수집하기", "수정하기", "복습하기", "실천하기", "실행하기", "마케팅하기", "연락하기",
"연구하기", "실습하기", "블로그 작성하기", "독서하기", "스터디 모임 참석하기", "영상보기", "목표 세우기",
"발표 자료 만들기", "가이드라인 작성하기", "일지 작성하기", "강의 만들기", "피트니스 챌린지하기",
"알고리즘 문제 풀기", "토론하기", "온라인 코스 수강하기", "포트폴리오 만들기", "데이터 분석하기",
"코드 최적화하기", "자동화 스크립트 만들기", "레포트 작성하기", "프레젠테이션 준비하기", "타임라인 설정하기",
"업데이트하기", "테크 블로그 읽기", "스프린트 회의하기", "배포 준비하기", "디버깅하기", "연습 세션하기",
"주제 발표하기", "진행상황 보고하기", "마인드맵 작성하기", "문제점 파악하기", "문제 해결 브레인스토밍하기",
"업데이트 테스트하기", "작업 분배하기", "이슈 분석하기", "지식 공유 세션 하기", "독서 기록하기",
"논문 읽기", "목표 달성 여부 확인하기", "운동 기록하기", "일정 관리하기", "새로운 기술 실습하기",
"워크샵 참여하기", "스터디 리드하기", "멘토링 하기", "협업 툴 사용하기", "데모 준비하기",
"시간 기록하기", "버전 관리하기", "토의 정리하기", "목표 리셋하기", "결과 리뷰하기", "필기 정리하기",
"소셜 미디어 관리하기", "발표 연습하기", "아이디어 구체화하기", "로그 분석하기", "설문 조사하기",
"사용자 피드백 받기", "성능 테스트하기", "미팅 스케줄하기", "문서화 작업하기", "워크플로우 개선하기",
"실습 결과 공유하기", "예산 관리하기", "자원 할당하기", "결과 공유하기", "비디오 튜토리얼 보기",
"실행 계획 짜기", "초안 작성하기", "피드백 문서화하기", "타겟 설정하기", "포스트잇 작업하기"


{format_instructions}
"""

def extract_activity_keywords_rag(name, desc, top_k=5):

    desc_embedding = embeddings.embed_query(desc)

    search_result = index.query(
        vector=desc_embedding,
        top_k=100,
        include_values=False,
        include_metadata=True,
        filter={
            'type': {'$eq': 'activity_keyword'}
        }
    )

    # 검색된 목표 키워드 및 메타데이터 수집
    retrieved_keywords = [match['metadata']['keyword'] for match in search_result['matches']]

    goal_context = '\n'.join(retrieved_keywords)

    prompt = PromptTemplate(
        input_variables=["desc", "name", "goal_context"],
        template=template,
        partial_variables={"format_instructions": format_instructions},
        output_keys=["desc", "name", "activityKeyword"]
    )

    extract_goal_chain = prompt | llm | activityKeyword_parser

    return extract_goal_chain.invoke(
        {
            "name": name,
            "desc": desc,
            "goal_context": goal_context
        }
    )


if __name__ == "__main__":
    # 사용자 입력
    planner_name = "카프카 기초"
    planner_desc = "카프카를 배우고 싶어"

    # 목표 키워드 추출
    activity_keyword = extract_activity_keywords_rag(planner_name, planner_desc)

    # 결과 출력
    print("플래너 이름:", planner_name)
    print("플래너 설명:", planner_desc)
    print("추출된 목표 키워드:")

    print(activity_keyword)
