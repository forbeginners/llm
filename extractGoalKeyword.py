import os

from dotenv import load_dotenv
import time

from langchain.retrievers import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from outputParser import goalKeyword_parser


load_dotenv()


def initialize_vectorstore():
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    index_name = 'keywords'

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

llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0.5
)

format_instructions = goalKeyword_parser.get_format_instructions()
template = """
Answer the question based on the following context:
{context}

플레너 이름: {name}
플레너 설명: {desc}

{format_instructions}
"""

prompt = PromptTemplate(
    input_variables=["desc", "name", "context"],
    template=template,
    partial_variables={"format_instructions": format_instructions},
    output_keys=["goalKeyword"],
)


vectorstore = initialize_vectorstore()
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 1,
        "fetch_k": 5,
        "lambda_mult": 0.5,
        'filter': {'type': 'goal_keyword'}
    },
)


def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])


retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever, llm=llm
)

extract_goal_chain = (
    {
        'context': retriever_from_llm | format_docs,
        'name': RunnablePassthrough(),
        'desc': RunnablePassthrough(),
    }
    | prompt
    | llm
    | goalKeyword_parser
)

# if __name__ == "__main__":
#     # 사용자 입력
#     planner_name = "카프카 기초"
#     planner_desc = "카프카를 배우고 싶어"
#
#     goalKeyword_keywords = extract_goal_chain.invoke({
#         'name': planner_name,
#         'desc': planner_desc,
#     })
#
#     # 결과 출력
#     print("플래너 이름:", planner_name)
#     print("플래너 설명:", planner_desc)
#     print("추출된 목표 키워드:")
#     print(goalKeyword_keywords)
