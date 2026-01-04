"""
混合搜索示例：使用 Milvus 进行稀疏向量和密集向量的混合检索
本示例演示如何使用 BGE-M3 模型生成稀疏和密集向量，并使用 RRF 融合器进行混合搜索
"""
# import os
# # hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_TOKEN'] = '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'
import json
import os
import numpy as np
from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, AnnSearchRequest, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 1. 初始化设置
COLLECTION_NAME = "dragon_hybrid_demo"  # 集合名称
MILVUS_URI = "http://localhost:19530"  # Milvus 服务器地址（服务器模式）
DATA_PATH = "../../data/C4/metadata/dragon.json"  # 数据文件路径（相对路径）
BATCH_SIZE = 50  # 批量插入数据的批次大小

# 2. 连接 Milvus 并初始化嵌入模型
print(f"--> 正在连接到 Milvus: {MILVUS_URI}")
# 连接到 Milvus 服务器，后续的 Collection数据操作都将使用此连接，也可以使用MilvusClient实例进行操作
# 参数:
#   uri: Milvus 服务器的 URI 地址
connections.connect(uri=MILVUS_URI)

print("--> 正在初始化 BGE-M3 嵌入模型...")
# 初始化 BGE-M3 嵌入函数，用于生成稀疏和密集向量
# 参数:
#   use_fp16: 是否使用 FP16 精度（False 表示使用 FP32）
#   device: 运行设备，可选 "cpu" 或 "cuda"
# 手动下载模型： hf download BAAI/bge-m3
ef = BGEM3EmbeddingFunction(
    # 使用本地已经下载好的模型
    model_name="/Users/kute/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
    use_fp16=False,
    device="cpu",
    model_kwargs={
        'use_safetensors': True  # 强制使用 safetensors
    }
)
print(f"--> 嵌入模型初始化完成。密集向量维度: {ef.dim['dense']}")

# 3. 创建 Collection
# 创建 Milvus 客户端实例
milvus_client = MilvusClient(uri=MILVUS_URI)

# 检查集合是否已存在，如果存在则删除
# 参数:
#   COLLECTION_NAME: 要检查的集合名称
if milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在删除已存在的 Collection '{COLLECTION_NAME}'...")
    # 删除已存在的集合
    # 参数:
    #   COLLECTION_NAME: 要删除的集合名称
    milvus_client.drop_collection(COLLECTION_NAME)

# 定义集合的字段模式
# FieldSchema 参数说明:
#   name: 字段名称
#   dtype: 数据类型（DataType 枚举）
#   is_primary: 是否为主键字段
#   auto_id: 是否自动生成 ID
#   max_length: 字符串类型的最大长度
#   dim: 向量维度（仅用于向量类型）
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),  # 主键字段，自动生成
    FieldSchema(name="img_id", dtype=DataType.VARCHAR, max_length=100),  # 图片 ID
    FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256),  # 图片路径
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),  # 标题
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),  # 描述信息
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),  # 类别
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),  # 位置
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),  # 环境
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),  # 稀疏向量字段
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"])  # 密集向量字段
]

# 如果集合不存在，则创建它及索引
if not milvus_client.has_collection(COLLECTION_NAME):
    print(f"--> 正在创建 Collection '{COLLECTION_NAME}'...")
    # 创建集合模式
    # CollectionSchema 参数:
    #   fields: 字段列表
    #   description: 集合描述
    schema = CollectionSchema(fields, description="关于龙的混合检索示例")

    # 创建集合
    # Collection 参数:
    #   name: 集合名称
    #   schema: 集合模式
    #   consistency_level: 一致性级别（Strong/Session/Bounded/Eventually）
    collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")
    print("--> Collection 创建成功。")

    # 4. 创建索引
    print("--> 正在为新集合创建索引...")
    # 稀疏向量索引配置
    # 参数:
    #   index_type: 索引类型，SPARSE_INVERTED_INDEX 适用于稀疏向量
    #   metric_type: 距离度量类型，IP 表示内积（Inner Product）
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    # 为稀疏向量字段创建索引
    # 参数:
    #   field_name: 字段名称
    #   index_params: 索引参数字典
    collection.create_index("sparse_vector", sparse_index)
    print("稀疏向量索引创建成功。")

    # 密集向量索引配置
    # 参数:
    #   index_type: 索引类型，AUTOINDEX 自动选择最佳索引
    #   metric_type: 距离度量类型，IP 表示内积
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    # 为密集向量字段创建索引
    collection.create_index("dense_vector", dense_index)
    print("密集向量索引创建成功。")

# 获取或创建集合对象，使用的是 connections.connect 的连接
collection = Collection(COLLECTION_NAME)

# 5. 加载数据并插入
# 将集合加载到内存中，以便进行搜索操作
collection.load()
print(f"--> Collection '{COLLECTION_NAME}' 已加载到内存。")

# 检查集合是否为空
if collection.is_empty:
    print(f"--> Collection 为空，开始插入数据...")
    # 检查数据文件是否存在
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")

    # 读取 JSON 数据文件
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 准备文档和元数据列表
    docs, metadata = [], []
    for item in dataset:
        # 将多个字段组合成一个文档字符串，用于生成向量
        parts = [
            item.get('title', ''),
            item.get('description', ''),
            item.get('location', ''),
            item.get('environment', ''),
            # 可选：添加更多字段
            # *item.get('combat_details', {}).get('combat_style', []),
            # *item.get('combat_details', {}).get('abilities_used', []),
            # item.get('scene_info', {}).get('time_of_day', '')
        ]
        # 过滤掉空字符串并用空格连接
        docs.append(' '.join(filter(None, parts)))
        metadata.append(item)
    print(f"--> 数据加载完成，共 {len(docs)} 条。")

    print("--> 正在生成向量嵌入...")
    # 使用 BGE-M3 模型同时生成稀疏和密集向量
    # 参数:
    #   docs: 文档列表
    # 返回:
    #   字典，包含 'sparse' 和 'dense' 两种向量
    # 这里会调用到 BGEM3EmbeddingFunction 的 __call__ 方法，入参为 docs 列表
    # 返回值为包含稀疏(sparse)和密集向量(dense)的字典
    embeddings = ef(docs)
    print("--> 向量生成完成。")

    print("--> 正在分批插入数据...")
    # 为每个字段准备批量数据
    img_ids = [doc["img_id"] for doc in metadata]
    paths = [doc["path"] for doc in metadata]
    titles = [doc["title"] for doc in metadata]
    descriptions = [doc["description"] for doc in metadata]
    categories = [doc["category"] for doc in metadata]
    locations = [doc["location"] for doc in metadata]
    environments = [doc["environment"] for doc in metadata]

    # 获取向量
    sparse_vectors = embeddings["sparse"]  # 稀疏向量
    dense_vectors = embeddings["dense"]  # 密集向量

    # 插入数据到集合
    # 参数:
    #   data: 数据列表，顺序必须与字段定义顺序一致（除了自动生成的主键）
    collection.insert([
        img_ids,
        paths,
        titles,
        descriptions,
        categories,
        locations,
        environments,
        sparse_vectors,
        dense_vectors
    ])

    # 刷新数据，确保数据持久化
    collection.flush()
    print(f"--> 数据插入完成，总数: {collection.num_entities}")
else:
    print(f"--> Collection 中已有 {collection.num_entities} 条数据，跳过插入。")

# 6. 执行搜索
search_query = "悬崖上的巨龙"  # 搜索查询文本
# search_filter是 Milvus 向量搜索中的过滤条件，用于在向量相似度搜索的基础上，根据标量字段进行数据过滤
# 过滤条件表达式，只搜索特定类别的文档
# 这里使用 Milvus 的表达式语法，过滤 category 字段
search_filter = 'category in ["western_dragon", "chinese_dragon", "movie_character"]'
top_k = 5  # 返回前 K 个最相似的结果

print(f"\n{'='*20} 开始混合搜索 {'='*20}")
print(f"查询: '{search_query}'")
print(f"过滤器: '{search_filter}'")

# 使用同一个 embeddings 模型 为查询文本生成向量
query_embeddings = ef([search_query])
dense_vec = query_embeddings["dense"][0]  # 获取密集向量
sparse_vec = query_embeddings["sparse"]._getrow(0)  # 获取稀疏向量（稀疏矩阵的第一行）

# 打印向量信息
print("\n=== 查询向量信息 ===")
print(f"查询-密集向量维度: {len(dense_vec)}")
print(f"查询-密集向量前5个元素: {dense_vec[:5]}")
print(f"查询-密集向量范数: {np.linalg.norm(dense_vec):.4f}")

print(f"\n查询-稀疏向量维度: {sparse_vec.shape[1]}")
print(f"查询-稀疏向量非零元素数量: {sparse_vec.nnz}")
print("查询-稀疏向量前5个非零元素:")
for i in range(min(5, sparse_vec.nnz)):
    print(f"  - 索引: {sparse_vec.indices[i]}, 值: {sparse_vec.data[i]:.4f}")
density = (sparse_vec.nnz / sparse_vec.shape[1] * 100)
print(f"\n稀疏向量密度: {density:.8f}%")

# search_params 是 Milvus 向量搜索中的搜索参数配置，用于控制向量索引的搜索行为和性能权衡
# 参数:
#   metric_type: 距离度量类型，IP 表示内积
#   params: 额外的搜索参数（此处为空）
search_params = {"metric_type": "IP", "params": {}}

# 先执行单独的搜索
print("\n--- [单独] 密集向量搜索结果 ---")
# 使用密集向量进行搜索
# collection.search 参数:
#   data: 查询向量列表
#   anns_field: 要搜索的向量字段名称
#   param: 搜索参数
#   limit: 返回结果数量
#   expr: 过滤表达式
#   output_fields: 需要返回的字段列表
dense_results = collection.search(
    [dense_vec],
    anns_field="dense_vector", # collection schema 中定义的密集向量字段名称
    param=search_params,
    limit=top_k,
    expr=search_filter,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

# 打印密集向量搜索结果
for i, hit in enumerate(dense_results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

print("\n--- [单独] 稀疏向量搜索结果 ---")
# 使用稀疏向量进行搜索
sparse_results = collection.search(
    [sparse_vec],
    anns_field="sparse_vector",
    param=search_params,
    limit=top_k,
    expr=search_filter,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

# 打印稀疏向量搜索结果
for i, hit in enumerate(sparse_results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

print("\n--- [混合] 稀疏+密集向量搜索结果 ---")
# 创建 RRF (Reciprocal Rank Fusion) 融合器
# RRFRanker 是 Reciprocal Rank Fusion（倒数排名融合） 重排序器，用于融合多个搜索结果列表，将它们合并成一个统一的排序结果。
# RRFRanker 参数:
#   k: RRF 算法的参数，用于平衡不同检索结果的排名，默认值通常为 60
#      k 值越小：对排名靠前的结果权重越大
#      k 值越大：对排名靠后的结果权重影响更大
rerank = RRFRanker(k=60)

# 创建搜索请求对象
# AnnSearchRequest 参数:
#   data: 查询向量列表
#   anns_field: 要搜索的向量字段名称
#   param: 搜索参数
#   limit: 返回结果数量
dense_req = AnnSearchRequest([dense_vec], "dense_vector", search_params, limit=top_k)
sparse_req = AnnSearchRequest([sparse_vec], "sparse_vector", search_params, limit=top_k)

# 执行混合搜索
# collection.hybrid_search 参数:
#   reqs: 搜索请求列表，包含多个 AnnSearchRequest 对象
#   rerank: 重排序器，用于融合多个搜索结果
#   limit: 最终返回的结果数量
#   output_fields: 需要返回的字段列表
results = collection.hybrid_search(
    reqs=[sparse_req, dense_req],
    rerank=rerank,
    limit=top_k,
    output_fields=["title", "path", "description", "category", "location", "environment"]
)[0]

# 打印最终混合搜索结果
for i, hit in enumerate(results):
    print(f"{i+1}. {hit.entity.get('title')} (Score: {hit.distance:.4f})")
    print(f"    路径: {hit.entity.get('path')}")
    print(f"    描述: {hit.entity.get('description')[:100]}...")

# 7. 清理资源
# 从内存中释放集合
# 参数:
#   collection_name: 要释放的集合名称
milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")

# 删除集合
# 参数:
#   collection_name: 要删除的集合名称
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")










