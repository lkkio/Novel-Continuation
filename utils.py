import os
import json
import re
import time
import random
import requests
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

TIKTOKEN_ENCODING = "cl100k_base"
CORRECTION_RULES = {
    "character_name_mapping": {
        "双清": "辛双清", "辛辛双清": "辛双清", "少女": "钟灵",
        "段公子": "段誉", "萧大王": "萧峰", "乔帮主": "萧峰", "琅擐": "琅嬛"
    },
    "force_names": ["神农帮", "无量剑", "灵鹫宫", "丐帮", "大理段氏", "逍遥派", "星宿派"],
    "empty_catchphrase_filler": "无",
    "default_era": "北宋哲宗年间",
    "character_specific_kungfu_clear": ["木婉清"]
}
API_RETRY_BASE_DELAY = 2
API_RETRY_MAX_DELAY = 60
API_RETRY_JITTER = 0.5

def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def check_env_vars():
    required_vars = ["API_KEY", "API_URL", "MODEL_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f" 缺失必要环境变量：{', '.join(missing_vars)}")
    print(" 环境变量校验通过")

def tiktoken_len(text):
    import tiktoken
    tokenizer = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return len(tokenizer.encode(text, disallowed_special=()))

def unify_and_correct(text):
    mapping = CORRECTION_RULES["character_name_mapping"]
    if isinstance(text, str):
        for old, new in mapping.items():
            text = text.replace(old, new)
    elif isinstance(text, list):
        text = [unify_and_correct(t) for t in text]
    elif isinstance(text, dict):
        text = {k: unify_and_correct(v) for k, v in text.items()}
    return text

def correct_characters(characters):
    if not isinstance(characters, list):
        return characters
    for char in characters:
        if not isinstance(char, dict):
            continue
        if char.get("name") in CORRECTION_RULES["character_specific_kungfu_clear"]:
            char["kungfu"] = []
        if not char.get("catchphrase"):
            char["catchphrase"] = CORRECTION_RULES["empty_catchphrase_filler"]
    return characters

def correct_world(world):
    if not isinstance(world, dict):
        world = {"era": CORRECTION_RULES["default_era"], "forces": [], "geography": [], "rules": []}
    if not world.get("era"):
        world["era"] = CORRECTION_RULES["default_era"]
    valid_geo = []
    invalid = CORRECTION_RULES["force_names"]
    if isinstance(world.get("geography"), list):
        valid_geo = [p for p in world.get("geography", []) if p not in invalid]
    world["geography"] = valid_geo
    return world

def deduplicate_kungfu(kungfu_list):
    if not isinstance(kungfu_list, list):
        return []
    kungfu_dict = {}
    for item in kungfu_list:
        if isinstance(item, dict) and "name" in item:
            name = item["name"]
            if name not in kungfu_dict:
                kungfu_dict[name] = item
            else:
                old_users = kungfu_dict[name].get("related_characters", [])
                new_users = item.get("related_characters", [])
                kungfu_dict[name]["related_characters"] = list(set(old_users + new_users))
                if len(item.get("description", "")) > len(kungfu_dict[name].get("description", "")):
                    kungfu_dict[name]["description"] = item["description"]
    return list(kungfu_dict.values())

def deduplicate_characters(characters):
    if not isinstance(characters, list):
        return []
    char_dict = {}
    for char in characters:
        if isinstance(char, dict) and "name" in char:
            name = char["name"]
            if name not in char_dict or len(char.get("experience", "")) > len(char_dict[name].get("experience", "")):
                char_dict[name] = char
    return list(char_dict.values())

def filter_non_human_characters(char_candidates):
    filtered = []
    force_names = CORRECTION_RULES["force_names"]
    place_keywords = ["山", "谷", "宫", "洞", "岛", "峰", "寺", "庄"]
    weapon_keywords = ["剑", "刀", "掌", "拳", "指", "棍", "鞭"]
    for char in char_candidates:
        if 2 <= len(char) <= 5 and char not in force_names:
            if not any(k in char for k in place_keywords + weapon_keywords):
                filtered.append(char)
    return filtered

def parse_json_result(result, expect_type="dict"):
    if not result:
        return None
    try:
        result = re.sub(r"```json|```", "", result).strip()
        if expect_type == "dict":
            s, e = result.find("{"), result.rfind("}")+1
        else:
            s, e = result.find("["), result.rfind("]")+1
        return json.loads(result[s:e]) if s != -1 else None
    except:
        return None

def call_llm_smart(system_prompt, user_prompt, temperature=0.3, max_tokens=2000):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('API_KEY')}"
    }
    data = {
        "model": os.getenv("MODEL_NAME"),
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": temperature, "max_tokens": max_tokens
    }
    for i in range(5):
        try:
            time.sleep(API_RETRY_BASE_DELAY)
            resp = requests.post(os.getenv("API_URL"), headers=headers, json=data, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait = min(API_RETRY_BASE_DELAY * (2**i), API_RETRY_MAX_DELAY)
                print(f" 限流等待 {wait}s")
                time.sleep(wait)
        except:
            time.sleep(3)
    return None

class VectorRetriever:
    def __init__(self, index, metadata, model):
        self.index = index
        self.metadata = metadata
        self.model = model  

    def search(self, query, top_k=5, return_fields=("chapter", "summary", "text")):
        if not self.index:
            return []

        emb = self.model.encode(query)
        emb = np.array([emb]).astype("float32")
        
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        scores, indices = self.index.search(emb, top_k)
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx >= len(self.metadata):
                continue
            item = {"score": round(float(scores[0][i]), 2)}
            for field in return_fields:
                item[field] = self.metadata[idx].get(field, "")
            results.append(item)
        return results

    def save(self, path):
        import faiss
        ensure_dir(path)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path, model_name):
        import faiss
        from sentence_transformers import SentenceTransformer
        
        print(f" 自动加载向量模型：{model_name}")
        model = SentenceTransformer(model_name)
        
        index_path = os.path.join(path, "index.faiss")
        meta_path = os.path.join(path, "metadata.json")
        index = faiss.read_index(index_path)
        
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        return cls(index, metadata, model)