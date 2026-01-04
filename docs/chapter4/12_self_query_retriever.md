# RAG 中 SelfQueryRetriever 的使用方式对比

## 核心区别总结

| 使用方式 | 适用场景 | 优势 |
|---------|---------|------|
| 直接调用 retriever | 简单检索测试 | 快速验证检索效果 |
| RetrievalQA Chain | 标准问答系统 | 自动组合检索+生成，简单易用 |
| LCEL 自定义链 | 需要灵活控制流程 | 高度可定制，现代化写法 |
| Conversational Chain | 多轮对话场景 | 支持上下文理解 |
| Agent 模式 | 复杂任务分解 | 可以组合多个工具和推理 |
## 1. 直接调用 Retriever

### 使用步骤
```python
# 1. 创建检索器
retriever = SelfQueryRetriever.from_llm(...)

# 2. 直接调用 invoke
results = retriever.invoke("查询问题")

# 3. 手动处理返回的文档
for doc in results:
    print(doc.page_content)
    print(doc.metadata)
```

### 适用场景
- **快速原型验证**：测试检索器配置是否正确
- **调试检索质量**：分析返回文档的相关性
- **批量文档检索**：需要获取原始文档而非生成答案
- **检索效果评估**：对比不同检索策略的召回率
- **数据探索**：浏览和分析向量库中的内容
- **集成到自定义流程**：在非 LangChain 框架中使用

### 优势
- **简单直接**：最少的代码量，无额外封装
- **完全控制**：可以自由处理检索结果
- **性能高效**：没有额外的 LLM 调用开销
- **灵活性强**：易于集成到任何工作流
- **调试友好**：可以直接查看检索到的原始文档
- **低成本**：不消耗 LLM API 调用额度

### 劣势
- 无法生成自然语言答案
- 需要手动实现答案合成逻辑
- 不适合面向终端用户的应用

---

## 2. RetrievalQA Chain

### 使用步骤
```python
# 1. 创建检索器
retriever = SelfQueryRetriever.from_llm(...)

# 2. 创建 QA 链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 或 map_reduce, refine
    retriever=retriever,
    return_source_documents=True
)

# 3. 提问并获取答案
result = qa_chain.invoke({"query": "用户问题"})
print(result['result'])  # 生成的答案
print(result['source_documents'])  # 来源文档
```

### 适用场景
- **标准问答系统**：企业知识库问答、文档查询系统
- **客服机器人**：基于文档回答用户咨询
- **技术文档助手**：根据官方文档回答技术问题
- **内容推荐系统**：基于元数据推荐视频、文章、商品
- **教育辅导**：根据教材内容回答学生问题
- **法律/医疗咨询**：从专业文档中提取答案
- **产品说明书助手**：帮助用户查询产品使用方法
- **研究文献综述**：自动总结相关论文内容

### 优势
- **开箱即用**：几行代码即可构建完整 RAG 系统
- **自动答案合成**：检索 + 生成一步完成
- **多种合成策略**：
  - `stuff`：适合文档较少，直接拼接
  - `map_reduce`：适合大量文档，分别总结后合并
  - `refine`：迭代式精炼答案
  - `map_rerank`：对多个答案打分排序
- **可追溯性**：自动返回来源文档
- **稳定可靠**：LangChain 官方维护，经过充分测试
- **易于上手**：适合初学者快速构建 MVP
- **配置灵活**：支持自定义提示词模板
- **错误处理完善**：内置异常捕获机制

### 劣势
- 灵活性相对较低
- 不支持多轮对话上下文
- 每次查询都是独立的

---

## 3. LCEL (LangChain Expression Language) 自定义链

### 使用步骤
```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 1. 创建检索器
retriever = SelfQueryRetriever.from_llm(...)

# 2. 定义格式化函数
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# 3. 使用管道操作符构建链
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. 执行链
answer = chain.invoke("用户问题")
```

### 适用场景
- **复杂业务逻辑**：需要多步骤处理的场景
- **自定义数据流**：精确控制数据在各组件间的传递
- **多模态 RAG**：结合文本、图片、表格等多种数据
- **混合检索策略**：同时使用多个检索器（向量+关键词+图谱）
- **条件分支逻辑**：根据查询类型选择不同处理路径
- **实时数据融合**：结合 API 调用、数据库查询和向量检索
- **答案后处理**：需要格式化、验证或增强生成内容
- **A/B 测试框架**：灵活切换不同的 Prompt 或模型
- **流式输出场景**：逐步展示生成过程
- **企业级应用**：需要精细控制和监控的生产环境

### 优势
- **极高灵活性**：可以任意组合和编排组件
- **现代化语法**：使用管道操作符 `|`，代码简洁优雅
- **类型安全**：支持完整的类型提示
- **并行处理**：可以同时执行多个检索或调用
- **条件路由**：根据输入动态选择执行路径
- **易于调试**：每个步骤都可以单独测试
- **性能优化**：支持批处理和异步执行
- **可观测性强**：方便添加日志和监控
- **可复用性**：链可以作为组件嵌套使用
- **渐进式迁移**：可以逐步从旧代码迁移

### 劣势
- 学习曲线较陡
- 需要更多代码量
- 对初学者不够友好

---

## 4. Conversational Retrieval Chain

### 使用步骤
```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 1. 创建检索器
retriever = SelfQueryRetriever.from_llm(...)

# 2. 创建对话记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# 3. 创建对话链
conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# 4. 多轮对话
response1 = conv_chain.invoke({"question": "第一个问题"})
response2 = conv_chain.invoke({"question": "后续问题"})  # 自动理解上下文
```

### 适用场景
- **智能客服系统**：需要理解用户多轮对话意图
- **个人助手应用**：如 ChatGPT 式的对话体验
- **教育辅导场景**：连续回答学生的追问
- **医疗问诊助手**：根据患者描述逐步细化问题
- **技术支持系统**：帮助用户逐步定位和解决问题
- **咨询顾问机器人**：提供连贯的专业建议
- **面试准备助手**：模拟多轮问答场景
- **法律咨询**：需要多轮澄清的复杂案件分析
- **产品推荐**：根据用户偏好逐步筛选
- **创意写作助手**：在对话中逐步完善内容

### 优势
- **上下文理解**：自动维护对话历史
- **指代消解**：理解"它"、"这个"等代词含义
- **连贯对话**：提供类似人类的交互体验
- **多种记忆策略**：
  - `ConversationBufferMemory`：保存完整历史
  - `ConversationSummaryMemory`：总结式记忆
  - `ConversationBufferWindowMemory`：滑动窗口
  - `ConversationKGMemory`：知识图谱记忆
- **自动查询改写**：将对话问题转换为独立查询
- **减少重复检索**：可以基于已有上下文回答
- **用户体验好**：更自然的交互方式
- **支持澄清提问**：AI 可以主动询问细节
- **适合复杂需求**：逐步引导用户表达真实需求

### 劣势
- 内存消耗随对话增长
- 长对话可能导致上下文混淆
- 需要合理的会话管理策略

---

## 5. Agent 模式

### 使用步骤
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool

# 1. 创建检索器
retriever = SelfQueryRetriever.from_llm(...)

# 2. 将检索器包装为工具
retriever_tool = create_retriever_tool(
    retriever,
    name="video_search",
    description="搜索B站视频信息的工具"
)

# 3. 准备工具列表（可以有多个工具）
tools = [retriever_tool, other_tool1, other_tool2]

# 4. 创建 Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    max_iterations=5
)

# 5. 执行复杂任务
result = agent_executor.invoke({
    "input": "复杂的多步骤问题"
})
```

### 适用场景
- **复杂任务分解**：需要多步推理和工具调用
- **混合信息源**：同时查询文档、API、数据库、网络
- **自主决策系统**：AI 自行判断使用哪些工具
- **数据分析助手**：查询数据 → 分析 → 可视化 → 报告
- **自动化工作流**：串联多个系统完成复杂任务
- **代码生成与执行**：检索示例 → 生成代码 → 测试 → 修复
- **研究助手**：文献检索 → 数据收集 → 分析 → 撰写
- **智能运维**：监控 → 日志分析 → 问题诊断 → 执行修复
- **电商购物助手**：搜索 → 比价 → 评价分析 → 推荐
- **旅行规划**：景点查询 → 路线规划 → 预订 → 提醒
- **投资研究**：新闻检索 → 财报分析 → 估值计算 → 报告生成

### 优势
- **最高灵活性**：可以组合任意数量和类型的工具
- **自主推理**：AI 自己决定调用哪个工具、调用顺序
- **处理复杂任务**：能分解和解决多步骤问题
- **可扩展性强**：轻松添加新工具和能力
- **工具组合能力**：一个查询可能触发多个工具调用
- **错误恢复**：可以根据结果调整策略
- **支持多种 Agent 类型**：
  - ReAct：推理-行动循环
  - Plan-and-Execute：先规划后执行
  - OpenAI Functions：函数调用型
  - Structured Chat：结构化对话型
- **透明的思维过程**：可以看到 AI 的推理步骤
- **动态适应**：根据中间结果调整后续行为
- **接近 AGI 的使用方式**：最智能的交互模式

### 劣势
- 最高的复杂度
- 需要精心设计工具描述
- 可能产生意外的工具调用序列
- Token 消耗较大
- 响应时间较长
- 需要更多的提示工程经验

---

## 选择建议流程图

```
开始
  ↓
是否需要生成答案？
  ├─ 否 → 直接调用 Retriever
  ↓
是否需要多轮对话？
  ├─ 是 → Conversational Chain
  ↓
是否需要复杂的自定义逻辑？
  ├─ 是 → 是否需要调用多个外部工具？
  │        ├─ 是 → Agent 模式
  │        └─ 否 → LCEL 自定义链
  ↓
否（简单问答）→ RetrievalQA Chain
```

## 实际项目中的组合使用

在真实项目中，往往需要组合使用多种模式：

**示例：企业智能客服系统架构**
```
用户问题
  ↓
【Agent 层】判断问题类型
  ├─ 简单FAQ → RetrievalQA（快速响应）
  ├─ 复杂咨询 → Conversational Chain（多轮对话）
  ├─ 需要操作 → LCEL 自定义链（调用业务API）
  └─ 数据查询 → 直接 Retriever（返回原始数据）
```

这样的分层架构能够：
- 提高响应速度（简单问题快速响应）
- 降低成本（按需使用 LLM）
- 提升用户体验（根据场景选择最佳模式）
- 便于维护和扩展