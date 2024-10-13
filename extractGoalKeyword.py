import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from outputParser import goalKeyword_parser

load_dotenv()

pc = Pinecone(
    api_key='6b199c8b-68e0-45a2-bc74-2eee73fb3ac1'
)
index = pc.Index('keywords')

llm = ChatOpenAI(
    model=os.environ['GOAL_MODEL'],
    temperature=os.environ['GOAL_TEMP']
)

format_instructions = goalKeyword_parser.get_format_instructions()

template = """
질문: 다음은 플레너의 이름과 플레너의 설명입니다. 이름과 설명을 읽고 해당 플레너의 목표 키워드들을 제공된 단어에서 10개 추출하세요.
목표 키워드란 플레너의 목적이 되는 키워드입니다.

플레너 이름: {name}
플레너 설명: {desc}

"배우다", "학습하다", "익히다", "훈련하다", "연습하다", "숙달하다", "조사하다", "탐구하다", "개선하다",
"증진하다", "준비하다", "계획하다", "성취하다", "완성하다", "달성하다", "관리하다", "유지하다",
"기록하다", "분석하다", "최적화하다", "개발하다", "실천하다", "실험하다", "도전하다", "강화하다",
"조직하다", "노트하다", "기록하다", "테스트하다", "복습하다", "요약하다", "정리하다", "실행하다",
"적용하다", "연구하다", "발표하다", "설계하다", "설정하다", "지도하다", "검토하다", "평가하다",
"진단하다", "구체화하다", "문제 해결하다", "혁신하다", "창조하다", "추구하다", "의사 결정하다",
"정보 수집하다", "업데이트하다", "모니터하다", "대응하다", "연계하다", "시스템화하다", "개선책 찾다",
"문서화하다", "주도하다", "효율화하다", "성공하다", "업무 분담하다", "동기부여하다", "통합하다",
"세부화하다", "직무 분석하다", "데이터 수집하다", "관찰하다", "예측하다", "분배하다", "피드백 받다",
"새로운 기술 습득하다", "정보 분석하다", "전략을 수립하다", "가이드라인 마련하다", "네트워킹하다",
"타임라인 정하다", "의견 조율하다", "우선순위 정하다", "목표 설정하다", "팀워크 강화하다",
"역량 강화하다", "질문하다", "해결 방안 찾다", "이론 적용하다", "통계 분석하다", "데이터 정제하다",
"실제 상황에 대처하다", "자원 관리하다", "협업하다", "조정하다", "성장하다", "경험 쌓다",
"문제 해결법 찾다", "전달하다", "기획하다", "성장 전략 세우다", "교정하다", "적응하다",
"시나리오 작성하다", "영향 분석하다", "계획 검토하다", "팀 구성하다", "리스크 분석하다", "혁신 주도하다"

{format_instructions}
"""
def extract_goal_keywords_rag(name, desc, top_k=5):
    # 2.1 검색 단계: Pinecone에서 관련 목표 키워드 검색
    embeddings = OpenAIEmbeddings()
    desc_embedding = embeddings.embed_query(desc)

    search_result = index.query(
        vector=desc_embedding,
        top_k=10,
        include_values=False,
        include_metadata=True,
        filter={
            'type': {'$eq': 'goal_keyword'}
        }
    )

    # 검색된 목표 키워드 및 메타데이터 수집
    retrieved_keywords = [match['metadata']['keyword'] for match in search_result['matches']]
    print(search_result)
    goal_context = '\n'.join(retrieved_keywords)

    print(retrieved_keywords)
    prompt = PromptTemplate(
        input_variables=["desc", "name", "goal_context"],
        template=template,
        partial_variables={"format_instructions": format_instructions},
        output_keys=["desc", "name", "goalKeyword"]
    )

    extract_goal_chain = prompt | llm | goalKeyword_parser

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
    planner_desc = "카프카 기초를 배웁니다. Spring과 Java를 활용합니다."

    # 목표 키워드 추출
    goal_keywords = extract_goal_keywords_rag(planner_name, planner_desc)

    # 결과 출력
    print("플래너 이름:", planner_name)
    print("플래너 설명:", planner_desc)
    print("추출된 목표 키워드:")
    for idx, keyword in enumerate(goal_keywords, start=1):
        print(f"{idx}. {keyword}")
