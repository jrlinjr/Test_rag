import re
from pypdf import PdfReader

import ollama
from ollama import Client
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer 

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text() # 把當前頁面轉成純文字
        # 以下在做正規化
        if text :
            text = re.sub(r'(?<![。！？\n])\n', '', text)  
            text = re.sub(r'\s+', ' ', text) 
            full_text += text + "\n"

    return full_text


def split_text_into_chunk(text, chunk_size=500):
    chunks = []
    total_length = len(text)

    for i in range(0,total_length,chunk_size):
        chunk_content = text[i : i + chunk_size]

        chunks.append(chunk_content)

    return chunks

def index_chunks_to_qdrant(chunks, collection_name="Test_Rag"):
    """
    使用 BAAI/bge-m3 模型將文字轉向量並存入 Qdrant
    """
    # device='mps'
    embedding_model = SentenceTransformer('BAAI/bge-m3') 
    client = QdrantClient(host="localhost", port=6333)
    vector_size = 1024 
    
    # 檢查並建立集合
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"已建立新集合：{collection_name} (維度: {vector_size})")
    
    points = []

    for i, chunk_text in enumerate(chunks):
        embedding_vector = embedding_model.encode(chunk_text).tolist()
        point = PointStruct(
            id=i,
            vector=embedding_vector,
            payload={"page_content": chunk_text}
        )
        points.append(point)

    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )


def search_qdrant(query, collection_name="Test_Rag"):
    """
    輸入問題，從 Qdrant 搜尋最相關的文本
    """    
    model = SentenceTransformer('BAAI/bge-m3')
    query_vector = model.encode(query).tolist()
    client = QdrantClient(host="localhost", port=6333)

    search_result = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=3
    ).points
  
    return search_result


def generate_answer_remote(query, search_result, remote_ip="http://10.0.0.209:11434"):    
    target_model = 'gpt-oss:20b' 
    client = Client(host=remote_ip)

    context_str = ""
    for result in search_result:
        context_str += result.payload['page_content'] + "\n---\n"

    final_prompt = f"""
    你是一個專業的助手。請「嚴格根據」以下提供的背景資料來回答使用者的問題。
    如果背景資料中沒有答案，請直接說明「資料不足，無法回答」，不要編造內容。

    【背景資料】：
    {context_str}

    【使用者問題】：
    {query}
    """
    
    # 發送請求
    response = client.chat(
        model=target_model,
        messages=[
            {'role': 'user', 'content': final_prompt},
        ],
        options={'temperature': 0.4}
    )

    return response['message']['content']        


# 主程式為以下
if __name__ == "__main__":
    pdf_filename = "/Users/jr/Downloads/離校注意事項.pdf"    
    
    raw_text = extract_text_from_pdf(pdf_filename)
    print(f"1. 文字讀取完成，共 {len(raw_text)} 字。")
    
    chunks = split_text_into_chunk(raw_text, chunk_size=500)
    print(f"2. 文字切分完成，共產生 {len(chunks)} 個區塊。")
    
    index_chunks_to_qdrant(chunks, collection_name="Test_Rag")
    print(f"3. 準備處理{len(chunks)}個區塊....")

    user_question = "哪些日子無法辦理離校？" 
    search_qdrant(user_question)
    print(f"問題為：{user_question}?")

    retrieved_docs = search_qdrant(user_question)

    if retrieved_docs:
        remote_ollama_ip = "http://10.0.0.209:11434" 
        
        answer = generate_answer_remote(user_question, retrieved_docs, remote_ollama_ip)
        
        print("\n" + "="*30)
        print("AI 的最終回答：")
        print(answer)
        print("="*30)
    else:
        print("沒有找到相關資料，跳過生成步驟。")