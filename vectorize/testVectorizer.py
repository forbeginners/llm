# 필요한 라이브러리 설치
# pip install openai pinecone-client

import os

from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import uuid

from Categories import data

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
    # index_name = 'keyword'
    # embeddings = OpenAIEmbeddings()
    # return Pinecone.from_existing_index(index_name, embeddings), embeddings
    index_name = 'keywords'
    embeddings = OpenAIEmbeddings()

    # Pinecone 초기화
    pc = Pinecone(
        api_key='6b199c8b-68e0-45a2-bc74-2eee73fb3ac1'
    )

    # 인덱스 존재 여부 확인 후 생성 또는 연결
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI Embeddings의 임베딩 차원 수
            metric='cosine'
        )
    # 인덱스에 연결

    index = pc.Index(index_name)

    return index, embeddings

def generate_uuid_id():
    return str(uuid.uuid4())
def store_goal_keywords(keywords):
    index, embeddings = initialize_vectorstore()

    # 업로드할 벡터 데이터 리스트 초기화
    vectors = []

    for item in keywords:
        keyword = item["keyword"]
        keyword_type = item["type"]

        # 임베딩 생성
        vector = embeddings.embed_query(keyword)

        # UUID 벡터 ID 생성
        vector_id = str(uuid.uuid4())

        # 벡터 데이터 구성
        vectors.append({
            'id': vector_id,
            'values': vector,
            'metadata': {
                'type': keyword_type,
                'keyword': keyword
            }
        })

        # Pinecone에 업로드 (배치 처리 권장)
    batch_size = 100  # 필요에 따라 조정
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"{i + len(batch)}개의 키워드를 저장했습니다.")

    print("모든 키워드가 Pinecone에 저장되었습니다.")


# 5. 함수 실행하여 키워드 저장
store_goal_keywords(all_keywords)