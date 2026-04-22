import os
import json
import random
import re
import argparse
from dotenv import load_dotenv
from utils import *
from prompt_template import Stage1Prompts

OUTPUT_DIR = "outputs"
ensure_dir(OUTPUT_DIR)

#文本预处理
def preprocess_novel(novel_path):
    with open(novel_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    if not raw.startswith('\n'):
        raw = '\n' + raw
    pattern = re.compile(r'\n(第[一二三四五六七八九十百千\d]+[章回][^\n]*)')
    matches = list(pattern.finditer(raw))
    chunks = []
    chapters = []
    if matches:
        for i, match in enumerate(matches):
            title = match.group(1).strip()
            start = match.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(raw)
            content = raw[start:end]
            clean = "\n".join([line.strip() for line in content.splitlines() if line.strip()])
            if not clean:
                continue
            # 二次切分过长章节
            if tiktoken_len(clean) < 5000:
                chunks.append(clean)
                chapters.append(title)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4000, chunk_overlap=500, length_function=tiktoken_len
                )
                subs = splitter.split_text(clean)
                for j, s in enumerate(subs):
                    chunks.append(s)
                    chapters.append(f"{title}-{j+1}")
    else:
        print("未检测到章节标记，将全文按固定长度切分")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=500, length_function=tiktoken_len
        )
        chunks = splitter.split_text(raw)
        chapters = [f"片段{i+1}" for i in range(len(chunks))]
    print(f"预处理完成：{len(chunks)} 个片段，首章节：{chapters[0] if chapters else '无'}")
    return chunks, chapters

#生成每个文本片段的局部摘要
def generate_summaries(chunks, chapters):
    sums = []
    sys_prompt = Stage1Prompts.SUMMARY
    
    for i, (c, ch) in enumerate(zip(chunks, chapters)):
        print(f"摘要 {i+1}/{len(chunks)} | {ch}")
        res = call_llm_smart(sys_prompt, f"请处理以下小说片段：\n{c[:3000]}", max_tokens=500)
        data = parse_json_result(res, "dict")
        if not data:
            data = {"core_plot": "", "characters": [], "key_scenes": []}
        data.update({"chunk_id": i, "chapter": ch})
        sums.append(data)
    with open(f"{OUTPUT_DIR}/local_summaries.json", "w", encoding="utf-8") as f:
        json.dump(sums, f, ensure_ascii=False, indent=2)
    return sums

#从局部摘要中统计人物出现频次，筛选出高频人物
def get_high_freq_chars(sums):
    count = {}
    for s in sums:
        for c in s.get("characters", []):
            count[c] = count.get(c, 0) + 1
    candidates = [k for k, v in sorted(count.items(), key=lambda x: x[1], reverse=True)]
    return filter_non_human_characters(candidates)

#生成整部小说的全局摘要
def gen_global(sums):
    sample = sums[:80] + sums[-40:]
    text = "\n".join([f"[{s['chapter']}] {s.get('core_plot', '')}" for s in sample])
    sys = Stage1Prompts.GLOBAL_SUMMARY
    
    res = call_llm_smart(sys, text, max_tokens=1500)
    data = parse_json_result(res, "dict") or {"main_plot": "", "ending_status": ""}
    with open(f"{OUTPUT_DIR}/global_summary.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data

# 人物库提取
def extract_characters(sums, chunks, freq):
    print("提取人物库")
    sample_text = ""
    for char in freq[:25]:
        for s in sums:
            if char in s.get("characters", []) and len(sample_text) < 10000:
                sample_text += chunks[s["chunk_id"]][:2000] + "\n"
    
    sys = Stage1Prompts.EXTRACT_CHARACTERS
    
    res = call_llm_smart(sys, sample_text[:12000], max_tokens=8000)
    data = parse_json_result(res, "list") or []
    data = deduplicate_characters(unify_and_correct(correct_characters(data)))
    with open(f"{OUTPUT_DIR}/characters.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"人物库：{len(data)} 人")
    return data

# 武功库提取
def extract_kungfu(chunks):
    print("提取武功库")
    indices = random.sample(range(len(chunks)), min(30, len(chunks)))
    text = "\n".join([chunks[i][:3000] for i in indices])
    
    sys = Stage1Prompts.EXTRACT_KUNGFU
    
    res = call_llm_smart(sys, text[:12000], max_tokens=8000)
    data = parse_json_result(res, "list") or []
    data = deduplicate_kungfu(data)
    with open(f"{OUTPUT_DIR}/kungfu.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"武功库：{len(data)} 个")
    return data

# 提取世界观
def extract_world(sums):
    print("提取世界观库")
    text = "\n".join([f"[{s['chapter']}] {s.get('core_plot', '')}" for s in random.sample(sums, min(20, len(sums)))])
    
    sys = Stage1Prompts.EXTRACT_WORLD
    
    res = call_llm_smart(sys, text, max_tokens=3000)
    data = parse_json_result(res, "dict") or {}
    data = correct_world(data)
    with open(f"{OUTPUT_DIR}/world.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data

# 写作风格提取
def extract_style(chunks):
    print("提取风格样本")
    indices = random.sample(range(len(chunks)), min(10, len(chunks)))
    text = "\n".join([chunks[i] for i in indices])

    sys = Stage1Prompts.EXTRACT_STYLE
    
    res = call_llm_smart(sys, text[:12000], max_tokens=3000)
    data = parse_json_result(res, "dict") or {}
    with open(f"{OUTPUT_DIR}/style_samples.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data

def main():
    parser = argparse.ArgumentParser(description="小说知识库构建工具")
    parser.add_argument("--novel", type=str, default="data/tianlongbabu.txt", help="小说TXT文件路径")
    args = parser.parse_args()
    load_dotenv()
    check_env_vars()
    print(f"开始构建知识库：{args.novel}")
    chunks, chapters = preprocess_novel(args.novel)
    # 保存原始数据供向量库使用
    with open(f"{OUTPUT_DIR}/chunks.json", "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "chapters": chapters}, f, ensure_ascii=False, indent=2)
    local_sums = generate_summaries(chunks, chapters)
    freq_chars = get_high_freq_chars(local_sums)
    gen_global(local_sums)
    extract_characters(local_sums, chunks, freq_chars)
    extract_world(local_sums)
    extract_kungfu(chunks)
    extract_style(chunks)
    print("知识库构建完成！所有文件保存在 outputs/ 目录")

if __name__ == "__main__":
    main()