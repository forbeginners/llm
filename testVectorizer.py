# 필요한 라이브러리 설치
# pip install openai pinecone-client

import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import uuid

from flask.Categories import data

load_dotenv()
# OpenAI 및 Pinecone API 키 설정


all_keywords = []

# 각 카테고리를 순회하며 키워드와 메타데이터를 추출
for keyword in data['goal_keywords']:
    all_keywords.append({
        "keyword": keyword,
        "type": "goal_keyword"
    })
# 활동 키워드 처리
for keyword in data['activity_keywords']:
    all_keywords.append({
        "keyword": keyword,
        "type": "activity_keyword"
    })
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

vectorstore = initialize_vectorstore()

def generate_uuid_id():
    return str(uuid.uuid4())

def store_keywords(keywords):
    # 업로드할 벡터 데이터 리스트 초기화
    vectors = []
    ids = []

    for item in keywords:
        keyword = item["keyword"]
        keyword_type = item["type"]

        # 임베딩 생성 벡터 ID 생성
        vector_id = str(uuid.uuid4())

        # 벡터 데이터 구성
        vectors.append(
            Document(page_content=keyword, metadata={'type':keyword_type})
        )
        ids.append(vector_id)

    vectorstore.add_documents(documents=vectors, ids=ids)

# 5. 함수 실행하여 키워드 저장
store_goal_keywords(all_keywords)