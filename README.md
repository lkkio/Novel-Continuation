# 多Agent小说续写框架（《天龙八部》专用）
基于Prompt驱动开发（Vibe Coding） + 多Agent协同的武侠小说续写工具，严格贴合原著人设、世界观与金庸文风，自动生成合规原创章节。

## 设计思路
### 阶段1：构建原著记忆库
把非结构化的原著文本，拆解成大模型可精准调用的结构化数据，从根源解决续写时人设跑偏、世界观混乱、剧情记混的问题：
1. 文本预处理：按章节 + 字数智能切分原著
2. 剧情摘要生成：生成片段级局部摘要 + 全书全局摘要
3. 核心元素提取：提取人物库、武功库、世界观库、文风样本库

### 阶段2：多Agent协同续写
将续写任务拆解为专业分工的多个 Agent，分工协作+生成-校验闭环，保证质量：
- 协调Agent：统筹流程、接收用户需求
- 大纲Agent：按起承转合生成合规剧情大纲
- 大纲校验Agent：检查人设/世界观是否合规
- 正文Agent：分段落生成金庸风格正文
- 正文校验Agent：校准细节，确保文风统一

## 快速运行
### 1. 安装依赖
```bash
pip install torch transformers sentence-transformers faiss-cpu python-dotenv numpy tqdm jieba
```

### 2. 配置模型接口
新建 `.env` 文件，填写：
```env
API_KEY=你的API密钥
API_URL=接口地址
MODEL_NAME=模型名称
```

### 3. 准备小说
将《天龙八部》txt放入 `data/tianlongbabu.txt`

### 4. 构建记忆库（仅需一次）
```bash
python stage1_main.py
python stage1_embedding.py
```

### 5. 开始续写
```bash
python stage2.py
```
按提示输入：续写时间点、核心事件、涉及人物

## 项目结构
```
├── data/              # 原著小说存放
├── outputs/           # 记忆库/向量库/续写结果输出
├── stage1_main.py     # 构建原著记忆库
├── stage1_embedding.py # 构建向量检索库
├── stage2.py          # 多Agent续写主程序
├── utils.py           # 工具函数
├── prompt_template.py # 所有Prompt模板
└── .env               # 接口配置
```
