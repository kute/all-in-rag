import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import IndexNode, TextNode
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.zhipuai import ZhipuAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

load_dotenv()

# llama 配置模型
Settings.llm = ZhipuAI(model="glm-4-air", api_key="f93957363d5942318ff78e037039e129.s5OSWVvZZqLLnaIN")
# Settings.llm = DeepSeek(model="deepseek-chat", api_key="sk-xxxxxxxx")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 1.加载数据并为每个工作表创建查询引擎和摘要节点
excel_file = '../../data/C3/excel/movie.xlsx'
xls = pd.ExcelFile(excel_file)

df_query_engines = {} # 存储工作表名称到其查询引擎的映射
all_nodes = [] # 用于存储所有sheet 的摘要节点
all_nodes_details = {} # 用于存储所有sheet 的详细节点

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    # print(f"DataFrame shape: {df.shape}")  # 检查数据维度
    # print(df.head())
    
    # 为当前工作表（DataFrame）创建一个 PandasQueryEngine
    # 添加指令约束
    instruction_str = (
        "1. 在访问索引前先检查 DataFrame 的长度\n"
        "2. 使用 len(df) 确保索引在有效范围内\n"
        "3. 优先使用条件过滤而非直接索引访问\n"
        "4. 如果需要访问特定行，先用 df.shape[0] 检查行数"
    )
    query_engine = PandasQueryEngine(df=df, llm=Settings.llm, verbose=True)

    year = sheet_name.replace('年份_', '')
    # 为当前工作表创建一个详细数据节点（TextNode）
    detail_text = f"详细数据：{year}年共有{len(df)}部电影，包括电影名称、评分、导演等完整信息。\n{df.head(0).to_string()}"
    detail_node_index_id = f"{sheet_name}_detail"
    detail_node = TextNode(
        text=detail_text,
        index_id=detail_node_index_id
    )
    all_nodes_details[detail_node_index_id] = detail_node
    
    # 为当前工作表创建一个摘要节点（IndexNode）
    summary = f"这个表格包含了年份为 {year} 的电影信息，可以用来回答关于这一年电影的具体问题。"
    # 创建摘要节点：包含摘要内容 和 sheetName 作为 index_id
    node = IndexNode(
        text=summary,
        index_id=sheet_name,
        # obj=detail_node  # 关联到详细数据节点的 index_id
    )
    all_nodes.append(node)
    
    # 存储工作表名称到其查询引擎的映射
    df_query_engines[sheet_name] = query_engine

# 2. 创建顶层索引（只包含摘要节点）
vector_index = VectorStoreIndex(all_nodes)

# 3. 创建递归检索器
# 3.1 创建顶层检索器，用于在摘要节点中检索
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# 3.2 创建递归检索器
recursive_retriever = RecursiveRetriever(
    "vector", # 查询图的根ID,指定第一个要查询的检索器标识符 GitHub,作为递归检索的起始点
    retriever_dict={"vector": vector_retriever}, # ID到检索器的映射字典 GitHub,定义了每个节点ID对应的检索器对象,用于在递归过程中根据节点引用查找对应的检索器
    query_engine_dict=df_query_engines, # ID到查询引擎的映射字典 GitHub,类似retriever_dict但映射的是查询引擎,允许节点链接到完整的查询引擎而不仅是检索器。
    # node_dict=all_nodes_details, # 详细节点字典 GitHub,包含了所有详细数据节点的映射,用于在递归过程中访问和查询这些节点的内容
    verbose=True,
)

# 4. 创建查询引擎
query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

# 5. 执行查询
query = "1994年评分人数最少的电影是哪一部？"
print(f"查询: {query}")
response = query_engine.query(query)
print(f"回答: {response}")


# 添加 node_dict 后的完整查询过程:
# 场景1: 检索到 IndexNode 且有 query_engine_dict 匹配
#
# 用户查询: "2020年评分最高的电影是什么?"
# 第一层检索: 使用 retriever_dict["vector"] 在摘要节点中检索,找到"年份_2020"的 IndexNode
# 检查 index_id: 发现 index_id="年份_2020" 在 query_engine_dict 中存在
# 执行查询引擎: 调用 df_query_engines["年份_2020"] 的 PandasQueryEngine 执行查询
# 返回结果: 返回具体电影名称和评分
#
# 场景2: 检索到 IndexNode 但只有 node_dict 匹配（假设没有配置 query_engine_dict）
#
# 用户查询: "2020年有多少部电影?"
# 第一层检索: 使用 retriever_dict["vector"] 找到"年份_2020"的 IndexNode
# 检查 obj 或 index_id:
#
# 如果 IndexNode 有 obj 属性指向详细节点,直接返回该节点
# 否则用 index_id (如"年份_2020_detail") 在 node_dict 中查找
#
#
# 返回详细节点: 返回包含详细数据描述的 TextNode
# LLM 处理: 查询引擎用 LLM 基于详细节点文本生成答案
#
# 场景3: 多层递归（IndexNode 引用另一个 IndexNode）
#
# 用户查询: "近期高分电影有哪些?"
# 第一层: 用 retriever_dict["vector"] 找到一个父级 IndexNode
# 第二层: 该 IndexNode 的 index_id 指向另一个检索器(如果在 retriever_dict 中),继续检索
# 第三层: 最终找到的 IndexNode 的 index_id 在 query_engine_dict 或 node_dict 中匹配
# 执行: 调用对应的查询引擎或返回详细节点
#
# 优先级规则:
# 检索到 IndexNode 后的处理顺序:
#
# 优先: 如果 index_id 在 query_engine_dict 中存在 → 调用查询引擎
# 其次: 如果 index_id 在 retriever_dict 中存在 → 继续递归检索
# 最后: 如果 index_id 在 node_dict 中存在 或 IndexNode 有 obj 属性 → 返回详细节点
#
# node_dict 的典型使用场景:
#
# 文档切块: 摘要节点指向多个详细内容节点,实现从摘要到原文的跳转
# 层级文档: 章节摘要指向章节详细内容,支持多层文档结构
# 混合检索: 有些节点需要查询引擎(结构化数据),有些只需要返回文本节点(非结构化文档)