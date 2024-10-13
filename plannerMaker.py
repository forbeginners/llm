# planner.py
import os
import logging

from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from extractActivityKeyword import extract_activity_chain, activity_context
from extractGoalKeyword import extract_goal_chain, goal_context
from utils.outputParser import planner_parser

# 환경 변수
load_dotenv()

# LLM 설정
llm = ChatOpenAI(
    model=os.environ["PLANNER_MODEL"],
    temperature=os.environ["PLANNER_TEMP"]
)

#  플래너 파싱을 위한 지시 프롬프트 생성
format_instructions = planner_parser.get_format_instructions()

# 로깅
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)


# retrieval qa를 사용한 chain 생성
def initialize_vectorstore():
    index_name = os.environ["PINECONE_INDEX"]
    embeddings = OpenAIEmbeddings()
    return Pinecone.from_existing_index(index_name, embeddings)


# 플래너 생성 프롬프트 템플릿
planner_template = """
    당신은 플래너를 만드는 전문가입니다. 사용자가 설명한 목표를 달성할 수 있도록 기본적으로 21일 동안 매일 수행할 수 있는 최소 21개의 할 일 목록을 작성하세요. 
    사용자가 목표를 완전히 달성하려면 21개 이상의 할 일이 필요할 수 있습니다. 이 경우, 21일 이상의 기간을 목표로 추가적인 할 일도 포함하세요.

    할 일 목록은 사용자가 21일 동안 매일 하나씩 수행할 수 있는 작은 습관을 형성하는 것을 목표로 합니다. 
    할 일들은 사용자의 설명과 태그에 맞춰 작성되며, 할 일은 간단하고 실행 가능해야 합니다. 
    
    예를 들어, '매일 30분 운동하기', '아침에 물 한 잔 마시기'와 같은 간단한 작업부터 시작해 점차 난이도가 올라가는 구조로 작성합니다.
    
    다음 활동 키워드와 목표 키워드도 참고하여 생성하세요.
    
    passThrough는 플레너의 이름과 설명입니다.
    
    passThrough: {passThrough}
    활동 키워드: {activityKeyword}
    목표 키워드: {goalKeyword}
    
    {format_instructions}
"""


# 프롬프트 생성
planner_prompt = PromptTemplate(
    input_variables=["passThrough", "activityKeyword", "goalKeyword"],
    template=planner_template,
    partial_variables={"format_instructions": format_instructions}
)


#  사용자의 입력을 최종으로 넘기기 위한 체인
def pass_through_chain(inputs):
    return {"name": inputs["name"], "desc": inputs["desc"]}


# 3개의 체인을 동시에 수행
parallel_chain = RunnableParallel(
    activityKeyword=extract_activity_chain,
    goalKeyword=extract_goal_chain,
    passThrough=pass_through_chain
)


# 플레너 생성 체인
planner_chain = planner_prompt | llm


rag_chain = parallel_chain | planner_chain | planner_parser

result = rag_chain.invoke({
    "activity_context": activity_context,
    "goal_context": goal_context,
    "name": "카프카 배우기",
    "desc": "카프카를 배우고 싶은데, 어떤 순서로 배우면 좋을까?"
})

# 기본 체인
# planner_chain = planner_prompt | llm | planner_parser

