import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_experimental.text_splitter import SemanticChunker
# from langchain_huggingface import HuggingFaceEmbeddings # 旧版导入路径，已弃用
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 初始化 SemanticChunker
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile", # 也可以是 "standard_deviation", "interquartile", "gradient"
    breakpoint_threshold_amount=95, # 百分位数阈值设为95
)

loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
documents = loader.load()

docs = text_splitter.split_documents(documents)

print(f"文本被切分为 {len(docs)} 个块。\n")
print("--- 前2个块内容示例 ---")
for i, chunk in enumerate(docs[:2]):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}):\n"{chunk.page_content}"')
