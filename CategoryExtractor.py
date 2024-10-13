import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as pcRAG
from pinecone import Pinecone
import uuid
from outputParser import extract_categories_parser

load_dotenv()

llm = ChatOpenAI(
    model=os.environ['GOAL_MODEL'],
    temperature=os.environ['GOAL_TEMP']
)
def initialize_vectorstore():
    index_name = os.environ["PINECONE_INDEX"]
    embeddings = OpenAIEmbeddings()
    return Pinecone.from_existing_index(index_name, embeddings)

# def save_new_category_to_pinecone(new_category, vectorstore, embeddings):
#     # 벡터화된 카테고리를 Pinecone에 저장
#     vector = embeddings.embed_query(new_category)
#
#     vector_id = str(uuid.uuid4())
#     vectorstore.upsert([{
#         'id': vector_id,
#         'values': vector,
#         'metadata': {
#             'text': new_category
#         }
#     }])
#
#
# def check_if_category_exists(category, vectorstore, embeddings):
#     # 카테고리가 인덱스 안에 존재하는지 확인
#     vector = embeddings.embed_query(category)
#     result = vectorstore.query(vector=[vector], top_k=5)
#     # 가장 유사한 결과의 점수가 일정 임계값 이상인지 확인
#     if result['matches'] and result['matches'][0]['score'] > 0.9:  # 유사도가 0.8 이상이면 존재한다고 판단
#         return True
#     return False




format_instructions = extract_categories_parser.get_format_instructions()

template = """
플레너 설명에서 주제와 관련된 가능한 모든 카테고리를 추출하세요.

플레너 설명에 나온 명사, 동사, 주제 등을 바탕으로 각 항목이 어떤 카테고리에 속하는지 판단하고 가능한 모든 카테고리를 나열하세요. 카테고리는 주제, 목표, 활동 등을 포함할 수 있습니다.

플레너 이름: {name}
플레너 설명: {desc}

추출된 카테고리:
{format_instructions}
"""
vectorstore = initialize_vectorstore()

retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 10, 'fetch_k': 50}
)

prompt = PromptTemplate(
    input_variables=["desc", "name", "categories"],
    template=template,
    partial_variables={"format_instructions": format_instructions},
    output_keys=["desc", "name", "categories"]
)

category_extract_chain = prompt | llm | extract_categories_parser


if __name__ == "__main__":
    # 사용자 입력
    planner_name = "골프 기초"
    planner_desc = "골프를 배워보고 싶어"

    docs = retriever.get_relevant_documents(planner_desc)

    result = category_extract_chain.invoke(
        {
            "name": planner_name,
            "desc": planner_desc,
            "categories": docs
        }
    )