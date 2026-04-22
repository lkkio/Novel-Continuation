#构建向量库
import os
import json
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"

from utils import ensure_dir, VectorRetriever

OUTPUT_DIR = "outputs"
VECTOR_DB_PATH = f"{OUTPUT_DIR}/vector_db"
MODEL_NAME = "BAAI/bge-small-zh-v1.5" 
ensure_dir(VECTOR_DB_PATH)

def load_data():
    with open(f"{OUTPUT_DIR}/chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        chunks = data["chunks"]
        chapters = data["chapters"]
    with open(f"{OUTPUT_DIR}/local_summaries.json", "r", encoding="utf-8") as f:
        sums = json.load(f)
    sum_dict = {s["chunk_id"]: s for s in sums}
    return chunks, chapters, sum_dict

def build_vector():
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        print("构建向量数据库")

        chunks, chapters, sum_dict = load_data()
        model = SentenceTransformer(MODEL_NAME)

        docs = []
        for i, chunk in enumerate(chunks):
            s = sum_dict.get(i, {})
            doc_text = f"章节：{chapters[i]}\n摘要：{s.get('core_plot', '')}\n原文：{chunk[:1500]}"
            docs.append({
                "id": i,
                "chapter": chapters[i],
                "summary": s.get("core_plot", ""),
                "text": doc_text,
                "raw_chunk": chunk
            })

        texts = [d["text"] for d in docs]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.array(embeddings).astype("float32"))

        retriever = VectorRetriever(index, docs, model)
        retriever.save(VECTOR_DB_PATH)
        print("向量库构建完成")

        test_result = retriever.search("段誉", top_k=2)
        if test_result:
            print(f"测试检索：{test_result[0]['chapter']} | 相似度：{test_result[0]['score']:.2f}")

    except Exception as e:
        print(f"构建失败：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    build_vector()