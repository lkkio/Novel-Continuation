import os
import json
import re
from dotenv import load_dotenv
from prompt_template import Stage2Prompts
from utils import (
    call_llm_smart, parse_json_result,
    VectorRetriever, ensure_dir,
    CORRECTION_RULES
)

OUTPUT_DIR = "outputs"
VECTOR_DB_PATH = f"{OUTPUT_DIR}/vector_db"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
ORIGINAL_TOTAL_CHAPTERS = 50

#统计有效汉字数量，排除标点、空格、换行、英文数字
def count_chinese_chars(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return len(chinese_pattern.findall(text))

TOTAL_TARGET_WORDS = 3000
STRUCTURE_CONFIG = {
    "起": {
        "min_word": int(TOTAL_TARGET_WORDS * 0.13),
        "max_word": int(TOTAL_TARGET_WORDS * 0.20),
        "core_content": "锚定时间（原著结局后3年）、人物（段誉、虚竹），铺垫核心动机（比如段誉要赴雁门关祭萧峰）"
    },
    "承": {
        "min_word": int(TOTAL_TARGET_WORDS * 0.18),
        "max_word": int(TOTAL_TARGET_WORDS * 0.25),
        "core_content": "触发核心事件（比如无量山残卷被盗/现世，虚竹到访汇合段誉，二人决定一同前往探查）"
    },
    "转": {
        "min_word": int(TOTAL_TARGET_WORDS * 0.37),
        "max_word": int(TOTAL_TARGET_WORDS * 0.46),
        "core_content": "赶赴现场，与反派对峙，核心打斗（必须符合战力体系），揭开反派的身世与动机（必须符合天龙内核）"
    },
    "合": {
        "min_word": int(TOTAL_TARGET_WORDS * 0.23),
        "max_word": int(TOTAL_TARGET_WORDS * 0.30),
        "core_content": "解决核心冲突，反派放下执念/受到惩戒，回归人物主线，留下轻微伏笔（比如段延庆的暗中注视、丐帮的传信）"
    }
}

FORBIDDEN_CHARACTERS = Stage2Prompts.FORBIDDEN_CHARACTERS
DUANYU_RULES = Stage2Prompts.DUANYU_RULES
XUZHU_RULES = Stage2Prompts.XUZHU_RULES
WORLD_RULES = Stage2Prompts.WORLD_RULES
STYLE_RULES = Stage2Prompts.STYLE_RULES

ensure_dir(OUTPUT_DIR)


class StoryMemory:
    def __init__(self):
        self.global_summary = self._load_json("global_summary.json")
        self.characters = self._load_json("characters.json")
        self.world = self._load_json("world.json")
        self.kungfu = self._load_json("kungfu.json")
        self.style_samples = self._load_json("style_samples.json")
        # 预加载段誉、虚竹的完整人设
        self.duanyu_full = next((c for c in self.characters if c["name"] == "段誉"), None)
        self.xuzhu_full = next((c for c in self.characters if c["name"] == "虚竹"), None)
    
    def _load_json(self, filename):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    # 优先返回对应场景的样本
    def get_style_examples(self, scene_type="fight"):
        samples = self.style_samples.get("samples", [])
        if not samples:
            return "金庸武侠小说原文文风，半文半白，叙事沉稳，有景物烘托和心理描写"
        
        for s in samples:
            if scene_type in s:
                return json.dumps(s, ensure_ascii=False)
        return json.dumps(samples[0], ensure_ascii=False)

# 多Agent核心实现
class CoordinatorAgent:
    def __init__(self, memory, retriever):
        self.memory = memory
        self.retriever = retriever
        self.user_requirement = None
        self.final_outline = None
        self.section_contents = {}
    
    def get_user_input(self):
        print("\n" + "="*50)
        print("《天龙八部》原著续写")
        time_point = input("1. 续写时间点（例：原著结局后3年）：").strip() or "原著结局后3年"
        core_conflict = input("2. 核心冲突/事件（例：抢夺逍遥派残卷）：").strip()
        involved_chars = input("3. 涉及原著人物（例：段誉,虚竹）：").strip() or "段誉,虚竹"
        print("="*50 + "\n")
        self.user_requirement = {
            "time_point": time_point,
            "core_conflict": core_conflict,
            "involved_chars": [c.strip() for c in involved_chars.split(",") if c.strip()]
        }
        return self.user_requirement
    
    def run_full_flow(self):
        req = self.get_user_input()
        # 按照起承转合结构生成大纲
        while True:
            outline = OutlineAgent(self.memory, self.retriever).generate(req)
            print("\n【生成的续写大纲】\n" + outline + "\n")
            
            outline_check = OutlineVerifyAgent(self.memory).verify(outline, req)
            if "PASS" in outline_check:
                print("大纲通过原著合规性与起承转合结构校验！")
                self.final_outline = outline
                break
            print(f"大纲违规，修改意见：{outline_check}")
            input("按回车键重新生成合规大纲...")
        
        # 按起承转合分段生成正文+逐段强校验
        section_order = ["起", "承", "转", "合"]
        for section_name in section_order:
            print(f"\n{'='*60}")
            print(f"正在处理【{section_name}】部分")
            
            config = STRUCTURE_CONFIG[section_name]
            current_content = ""
            retry_count = 0
            max_retry = 5
            
            while retry_count < max_retry:
                if retry_count == 0:
                    # 第一次：全新生成
                    print(f"\n 第1次尝试：全新生成【{section_name}】...")
                    content = SectionWriterAgent(self.memory, self.retriever).write_single_section(
                        full_outline=self.final_outline,
                        section_name=section_name,
                        user_req=req,
                        last_retry_info=""
                    )
                else:
                    # 第N次：在原有基础上扩写
                    print(f"\n 第{retry_count+1}次尝试：在原有基础上扩写【{section_name}】...")
                    content = SectionWriterAgent(self.memory, self.retriever).expand_section(
                        current_content=current_content,
                        full_outline=self.final_outline,
                        section_name=section_name,
                        user_req=req,
                        last_retry_info=last_error_info
                    )
                
                # 精准统计汉字数
                actual_word = count_chinese_chars(content)
                print(f" 本次生成有效汉字数：{actual_word}（要求：{config['min_word']}-{config['max_word']}）")
                
                # 检查合规性
                section_check, error_details = SectionVerifyAgent(self.memory).verify_single_section(
                    content=content,
                    section_name=section_name,
                    full_outline=self.final_outline,
                    user_req=req,
                    min_word=config['min_word'],
                    max_word=config['max_word']
                )
                
                if section_check == "PASS":
                    print(f" 【{section_name}】通过所有校验！")
                    self.section_contents[section_name] = content
                    break
                
                # 不合格，准备重试
                current_content = content
                last_error_info = error_details
                print(f" 【{section_name}】需要优化：{error_details}")
                retry_count += 1
            
            if retry_count >= max_retry:
                print(f" 【{section_name}】多次尝试后仍未完全达标，但已接近要求，继续下一步...")
                self.section_contents[section_name] = current_content
        
        # 全文终审+生成回目+保存
        full_raw_content = "\n\n".join([self.section_contents[name] for name in section_order])
        total_word = count_chinese_chars(full_raw_content)
        print(f"\n 全文总有效汉字数：{total_word}")
        final_check = ContentVerifyAgent(self.memory).verify(full_raw_content, req)
        final_chapter = self._add_standard_chapter_title(full_raw_content, self.final_outline)
        self._save_result(final_chapter, self.final_outline, final_check, self.section_contents, total_word)
        return final_chapter
    
    def _add_standard_chapter_title(self, content, outline):
        sys_prompt = Stage2Prompts.CHAPTER_TITLE
        user_prompt = f"根据以下起承转合剧情大纲，生成《天龙八部》续写的第五十一回对仗回目标题\n大纲：{outline}"
        title = call_llm_smart(sys_prompt, user_prompt, temperature=0.5, max_tokens=100)
        if "第五十一回" not in title:
            title = title.replace("第", "第五十一回 ").replace("回", "")
        return f"{title}\n\n{content}"
    
    def _save_result(self, final_chapter, outline, check_report, section_contents, total_word):
        chapter_path = os.path.join(OUTPUT_DIR, "《天龙八部》续写.txt")
        with open(chapter_path, "w", encoding="utf-8") as f:
            f.write(final_chapter)
        
        process_path = os.path.join(OUTPUT_DIR, "续写过程.json")
        with open(process_path, "w", encoding="utf-8") as f:
            json.dump({
                "user_requirement": self.user_requirement,
                "plot_outline": outline,
                "section_contents": section_contents,
                "total_chinese_words": total_word,
                "structure": "起承转合",
                "final_check_report": check_report
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n续写完成！最终章节已保存至：{chapter_path}")
        print(f"过程记录已保存至：{process_path}")

class OutlineAgent:
    #大纲生成Agent：严格按照起承转合结构生成
    def __init__(self, memory, retriever):
        self.memory = memory
        self.retriever = retriever
    
    def generate(self, user_req):
        print("【大纲生成Agent】")
        # 召回段誉、虚竹、核心冲突的原著片段
        rag_query = f"{user_req['core_conflict']} 段誉 虚竹 原著经典情节"
        rag_context = self.retriever.search(rag_query, top_k=4)
        # 打印检索日志
        print(f" 向量库已召回{len(rag_context)}条原著相关片段：")
        for i, item in enumerate(rag_context):
            print(f"  {i+1}. 章节：{item['chapter']} | 摘要：{item['summary'][:60]}...")
        
        config = STRUCTURE_CONFIG
        system_prompt = Stage2Prompts.OUTLINE_AGENT.format(
            forbidden_chars=FORBIDDEN_CHARACTERS,
            duanyu_rules=DUANYU_RULES,
            xuzhu_rules=XUZHU_RULES,
            world_rules=WORLD_RULES,
            qi_min=config['起']['min_word'], qi_max=config['起']['max_word'],
            cheng_min=config['承']['min_word'], cheng_max=config['承']['max_word'],
            zhuan_min=config['转']['min_word'], zhuan_max=config['转']['max_word'],
            he_min=config['合']['min_word'], he_max=config['合']['max_word'],
            core_conflict=user_req['core_conflict']
        )
        
        user_prompt = f"""
【原著核心设定】
段誉完整人设：{json.dumps(self.memory.duanyu_full, ensure_ascii=False)}
虚竹完整人设：{json.dumps(self.memory.xuzhu_full, ensure_ascii=False)}
全书主线结局：{self.memory.global_summary.get('main_plot', '')} | {self.memory.global_summary.get('ending_status', '')}
时代背景与势力：{json.dumps(self.memory.world, ensure_ascii=False)}
【用户续写要求】时间点：{user_req['time_point']}，核心冲突：{user_req['core_conflict']}，涉及人物：{user_req['involved_chars']}
【RAG原著参考片段（必须参考人设与文风）】{json.dumps(rag_context, ensure_ascii=False)}
        """
        res = call_llm_smart(system_prompt, user_prompt, temperature=0.7, max_tokens=3500)
        parsed = parse_json_result(res, expect_type="dict")
        if parsed and "outline" in parsed:
            outline_text = "\n".join([
                f"【{p['part']}】\n情节：{p['content']}\n涉及人物：{','.join(p['involved_chars'])}\n场景：{p['scene']}\n"
                for p in parsed['outline']
            ])
            if "original_elements" in parsed:
                outline_text += f"\n【原创合规设定】\n{json.dumps(parsed['original_elements'], ensure_ascii=False, indent=2)}"
            return outline_text
        return res

class OutlineVerifyAgent:
    #大纲校验Agent
    def __init__(self, memory):
        self.memory = memory
    
    def verify(self, outline, user_req):
        system_prompt = Stage2Prompts.OUTLINE_VERIFY.format(forbidden_chars=FORBIDDEN_CHARACTERS)
        user_prompt = f"""
【待校验大纲】{outline}
【用户要求】{user_req}
        """
        return call_llm_smart(system_prompt, user_prompt, temperature=0.1, max_tokens=800)

class SectionWriterAgent:
    #正文生成Agent
    def __init__(self, memory, retriever):
        self.memory = memory
        self.retriever = retriever
    
    def write_single_section(self, full_outline, section_name, user_req, last_retry_info=""):
        # 【分场景精准检索】
        section_content = full_outline.split(f"【{section_name}】")[1].split("【")[0]
        rag_query = f"{section_name} {section_content} 段誉 虚竹 金庸文风 原著片段"
        rag_context = self.retriever.search(rag_query, top_k=3)
        print(f" 【{section_name}】向量库已召回{len(rag_context)}条原著片段")
        
        config = STRUCTURE_CONFIG[section_name]
        scene_type = "fight" if section_name == "转" else "scene"
        
        system_prompt = Stage2Prompts.SECTION_WRITER.format(
            min_word=config['min_word'],
            max_word=config['max_word'],
            section_name=section_name,
            core_content=config['core_content'],
            last_retry_info=last_retry_info,
            forbidden_chars=FORBIDDEN_CHARACTERS,
            duanyu_rules=DUANYU_RULES,
            xuzhu_rules=XUZHU_RULES,
            world_rules=WORLD_RULES,
            style_rules=STYLE_RULES,
            style_example=self.memory.get_style_examples(scene_type)
        )
        
        user_prompt = f"""
【完整起承转合大纲】{full_outline}
【当前必须续写的段落】【{section_name}】，本段核心情节：{section_content}
【RAG原著参考片段】{json.dumps(rag_context, ensure_ascii=False)}
【段誉完整人设】{json.dumps(self.memory.duanyu_full, ensure_ascii=False)}
【虚竹完整人设】{json.dumps(self.memory.xuzhu_full, ensure_ascii=False)}
        """
        return call_llm_smart(system_prompt, user_prompt, temperature=0.85, max_tokens=config['max_word']*2)
    
    def expand_section(self, current_content, full_outline, section_name, user_req, last_retry_info=""):
        #在原有基础上扩写内容
        section_content = full_outline.split(f"【{section_name}】")[1].split("【")[0]
        config = STRUCTURE_CONFIG[section_name]
        scene_type = "fight" if section_name == "转" else "scene"
        
        system_prompt = Stage2Prompts.SECTION_EXPANDER.format(
            min_word=config['min_word'],
            max_word=config['max_word'],
            core_content=config['core_content'],
            last_retry_info=last_retry_info,
            forbidden_chars=FORBIDDEN_CHARACTERS,
            duanyu_rules=DUANYU_RULES,
            xuzhu_rules=XUZHU_RULES,
            world_rules=WORLD_RULES,
            style_rules=STYLE_RULES,
            style_example=self.memory.get_style_examples(scene_type)
        )
        
        user_prompt = f"""
【完整起承转合大纲】{full_outline}
【当前必须扩写的段落】【{section_name}】，本段核心情节：{section_content}
【现有内容（需要扩写）】{current_content}
【段誉完整人设】{json.dumps(self.memory.duanyu_full, ensure_ascii=False)}
【虚竹完整人设】{json.dumps(self.memory.xuzhu_full, ensure_ascii=False)}
        """
        return call_llm_smart(system_prompt, user_prompt, temperature=0.85, max_tokens=config['max_word']*2)

class SectionVerifyAgent:
    #分段校验Agent
    def __init__(self, memory):
        self.memory = memory
    
    def verify_single_section(self, content, section_name, full_outline, user_req, min_word, max_word):
        print(f"【分段校验Agent】正在审核【{section_name}】...")
        actual_word = count_chinese_chars(content)
        
        system_prompt = Stage2Prompts.SECTION_VERIFY.format(section_name=section_name)
        
        user_prompt = f"""
【本段正文】{content}
【本段大纲】{full_outline.split(f"【{section_name}】")[1].split("【")[0]}
【当前有效汉字数】{actual_word}，要求字数：{min_word}-{max_word}字
        """
        
        res = call_llm_smart(system_prompt, user_prompt, temperature=0.1, max_tokens=500)
        parsed = parse_json_result(res, expect_type="dict")
        
        # 额外检查字数
        if actual_word < min_word:
            word_error = f"有效汉字数不足，要求最少{min_word}，实际只有{actual_word}。请在景物描写、心理活动、对话、打斗细节等地方增加内容。"
            if parsed:
                if parsed["check_result"] == "PASS":
                    return ("FAIL", word_error)
                else:
                    return ("FAIL", parsed["error_details"] + " " + word_error)
            else:
                return ("FAIL", word_error)
        
        if actual_word > max_word:
            word_error = f"有效汉字数超出，要求最多{max_word}，实际有{actual_word}。请适当精简。"
            if parsed:
                if parsed["check_result"] == "PASS":
                    return ("FAIL", word_error)
                else:
                    return ("FAIL", parsed["error_details"] + " " + word_error)
            else:
                return ("FAIL", word_error)
        
        if parsed:
            return (parsed["check_result"], parsed.get("error_details", ""))
        else:
            return ("PASS", "")

class ContentVerifyAgent:
    #全文终审Agent
    def __init__(self, memory):
        self.memory = memory
    
    def verify(self, content, user_req):
        print("【全文终审Agent】")
        total_word = count_chinese_chars(content)
        system_prompt = Stage2Prompts.CONTENT_VERIFY
        
        user_prompt = f"""
【续写全文】{content}
【用户要求】{user_req}
【全文总有效汉字数】{total_word}
        """
        return call_llm_smart(system_prompt, user_prompt, temperature=0.2, max_tokens=800)

if __name__ == "__main__":
    try:
        load_dotenv()
        # 加载记忆库与向量检索器
        story_memory = StoryMemory()
        vector_retriever = VectorRetriever.load(VECTOR_DB_PATH, EMBEDDING_MODEL)
        # 启动协调Agent
        coordinator = CoordinatorAgent(story_memory, vector_retriever)
        final_chapter = coordinator.run_full_flow()
        print("\n" + "="*50)
        print("【最终续写章节预览】")
        print(final_chapter[:2500] + "......\n")
    except Exception as e:
        print(f"\n 流程执行失败：{str(e)}")
        import traceback
        traceback.print_exc()