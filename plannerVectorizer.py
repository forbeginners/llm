# import os
# import logging
# import json
# import boto3
# import requests
#
# from dotenv import load_dotenv
# from langchain_community.vectorstores import Pinecone
# from langchain_openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
#
# load_dotenv()
#
# logging.basicConfig(
#     format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
# )
# logger = logging.getLogger(__name__)
# # 벡터화 준비
# def initialize_vectorstore():
#     index_name = os.environ["PINECONE_INDEX"]
#     embeddings = OpenAIEmbeddings()
#     return Pinecone.from_existing_index(index_name, embeddings)
# def fetch_data_from_mysql():
#
# def fetch_new_planners_from_es():
#     es_url=""
#
#     query = {
#         "query": {
#             "range": {
#                 "timestamp": {
#                     "gte": "now-1d/d",  # 지난 24시간의 데이터 가져오기 (필요에 따라 수정 가능)
#                 }
#             }
#         }
#     }
#
#     headers = {"Content-Type": "application/json"}
#     response = requests.get(es_url, headers=headers, data=json.dumps(query))
#
#     if response.status_code == 200:
#         return response.json()['hits']['hits']
#     else:
#         print(f"Elasticsearch fetch error: {response.status_code}")
#         return []
#
#
import os
import logging
import json
import boto3
import mysql.connector
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# 벡터화 준비
def initialize_vectorstore():
    index_name = os.environ["PINECONE_INDEX"]
    embeddings = OpenAIEmbeddings()
    return Pinecone.from_existing_index(index_name, embeddings)


# MySQL 데이터베이스 연결
def get_mysql_connection():
    return mysql.connector.connect(
        host=os.environ["MYSQL_HOST"],
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"],
    )


# MySQL에서 데이터를 가져와 Pinecone에 저장하는 함수
def fetch_data_from_mysql():
    try:
        connection = get_mysql_connection()
        cursor = connection.cursor(dictionary=True)
        # planner와 todo 데이터를 조인해서 가져오기
        query = """
        SELECT 
            p.planner_id, p.planner_name, p.planner_description, 
            GROUP_CONCAT(t.todo_name SEPARATOR ', ') AS todos
        FROM planner p
        LEFT JOIN todo t ON p.planner_id = t.planner_id
        GROUP BY p.planner_id;
        """
        cursor.execute(query)
        planners = cursor.fetchall()

        # Pinecone 벡터 스토어 초기화
        vectorstore = initialize_vectorstore()

        for planner in planners:
            # 플래너 정보와 TODO를 결합한 텍스트
            combined_text = f"Planner Name: {planner['planner_name']}. " \
                            f"Description: {planner['planner_description']}. " \
                            f"Todos: {planner['todos']}."
            logger.info(f"Processing planner ID {planner['planner_id']}")

            # 텍스트 분할 및 벡터화
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_text(combined_text)

            # 각 텍스트를 Pinecone에 벡터로 저장
            for text in texts:
                metadata = {"planner_id": planner["planner_id"]}
                vectorstore.add_texts([text], metadatas=[metadata])

        logger.info("Data successfully stored in Pinecone.")

    except mysql.connector.Error as err:
        logger.error(f"MySQL error: {err}")


if __name__ == "__main__":
    fetch_data_from_mysql()