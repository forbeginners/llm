# planner.py

import os
import time
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_community.vectorstores import Pinecone
from langchain.retrievers import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from extractActivityKeyword import extract_activity_chain
from extractGoalKeyword import extract_goal_chain
from outputParser import planner_parser

load_dotenv()

# LLM 설정
llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0.5
)


format_instructions = planner_parser.get_format_instructions()


# retrieval qa를 사용한 chain 생성
def initialize_vectorstore():
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    index_name = 'planners'

    # 인덱스 존재 확인
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())

    return vector_store


# 플래너 생성 프롬프트 템플릿
planner_template = """
    Answer the question based on the following context:
    {context}
    
    플래너 이름: {name}
    플래너 설명: {desc}
    목표 키워드: {goalKeyword}
    활동 키워드: {activityKeyword}
    목표 기간: {target_period} (예: 4 = 4주)
    반복 요일: {repeating_days} (예: [1, 3, 5] = 월, 수, 금)
    
    당신은 사용자의 목표 달성을 돕는 플래너 생성 도우미입니다. 
    아래 제공된 사용자 입력 정보를 고려하여, 사용자의 목표를 달성하기 위한 상세한 플래너를 생성해 주세요.
    
    플래너를 작성할 때 다음 두 가지 원칙을 반드시 지켜 주세요:

    SMART 원칙 준수:
        구체적 (Specific): 각 투두 항목은 명확하고 구체적으로 작성되어야 합니다.
        측정 가능 (Measurable): 진행 상황을 측정할 수 있는 요소를 포함해야 합니다.
        달성 가능 (Achievable): 현실적인 범위 내에서 달성 가능한 목표를 설정해야 합니다.
        관련성 있는 (Relevant): 목표 키워드와 활동 키워드에 직접적으로 연관되어야 합니다.
        시간 제한이 있는 (Time-bound): 명확한 시간 프레임이나 데드라인을 포함해야 합니다.
        
    투두 항목은 21개 이상 작성:
        전체 플래너 기간 동안 최소 21개의 투두 항목을 포함해 주세요.
        투두 항목은 반복 요일에 따라 고르게 분배되어야 합니다.
        습관 형성을 위해 지속적인 활동을 계획에 반영해 주세요.
    
    점진적인 난이도 향상:
        투두는 난이도가 향상하는 순서로 되어야 합니다. 
        만약 초보라면 초보 수준에서 가능한 투두 항목들을 선정하고, 난이도 순으로 나열하세요.
        
    예를 들어, 플래너 이름이 "카프카 기초 플래너"이고, 목표 키워드가 ["배우다", "이해하다", "익히다"], 활동 키워드가 ["공부하다", "실습하다", "복습하다"]인 경우, 다음과 같이 투두 항목을 작성할 수 있습니다:

        "카프카의 기본 개념을 공부하고 노트에 정리한다."
        "Java로 간단한 카프카 프로듀서와 컨슈머를 구현한다."
        "실습한 내용을 바탕으로 퀴즈를 풀어 이해도를 확인한다."
        
    이러한 방식으로 사용자에게 맞춤화된 플래너를 생성해 주세요.
    
    투두 형식은 "카프카의 기본 개념을 공부하고 노트에 정리한다."와 같이 내용만 포함해야합니다.

    
    {format_instructions}
"""

vectorstore = initialize_vectorstore()
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 5,
        "lambda_mult": 0.5,
    },
)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever, llm=llm
)

# 프롬프트 생성
planner_prompt = PromptTemplate(
    input_variables=['context', 'goalKeyword', 'activityKeyword', 'name', 'desc', "target_period", "repeating_days"],
    template=planner_template,
    partial_variables={"format_instructions": format_instructions}
)


#  사용자의 입력을 최종으로 넘기기 위한 체인
def name_pass_through(name):
    return name['name']
def desc_pass_through(desc):
    return desc['desc']
def repeat_day_pass_through(desc):
    return desc['repeating_days']
def target_period_pass_through(desc):
    return desc['target_period']
def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])


# 2개의 체인을 동시에 수행
parallel_chain = RunnableParallel(
    activityKeyword=extract_activity_chain,
    goalKeyword=extract_goal_chain,
    namePassThrough=name_pass_through,
    descPassThrough=desc_pass_through,
)


# 플레너 생성 체인
planner_chain = (
    {
        'context': retriever_from_llm | format_docs,
        'name': name_pass_through,
        'desc': desc_pass_through,
        'repeating_days': repeat_day_pass_through,
        'target_period': target_period_pass_through,
        'activityKeyword': extract_activity_chain,
        'goalKeyword': extract_goal_chain,
    }
    | planner_prompt
    | llm
)


rag_chain = planner_chain | planner_parser


# if __name__ == "__main__":
#     # 사용자 입력
#     planner_name = "클라이밍 기초"
#     planner_desc = "클라이밍을 배우고 싶어"
#     target_period = 5
#     repeating_days = '[1, 3, 5]'
#
#     # 목표 키워드 추출
#     planner = rag_chain.invoke({
#         'name': planner_name,
#         'desc': planner_desc,
#         'target_period': target_period,
#         'repeating_days': repeating_days
#     })
#
#     # 결과 출력
#     print("플래너 이름:", planner_name)
#     print("플래너 설명:", planner_desc)
#     print("planner:")
#     print(planner)
#




