"""第 4 课：RAG 的检索环节 —— 切分 / 向量化 / 检索 / 拼进提示词。

说明：
    DeepSeek 没有 embedding 接口，这里用 common.embeddings 里的**确定性假 embedding**
    （按字符哈希）来演示检索流程本身。换成真实向量库时，只需把 HashEmbeddings 换成
    OpenAIEmbeddings / DashScopeEmbeddings 等，其余代码不用动。

核心认知：
    RAG = 先检索、再生成。检索质量的瓶颈通常在**切分**和**检索条数**，
    而不是模型。

运行：uv run python -m core.l4_rag
"""

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.config import get_model
from common.embeddings import HashEmbeddings

DOC = """
LangGraph 是 LangChain 团队推出的低层编排框架，核心是「状态 + 图」。
图的节点是普通函数，边决定下一步走向，条件边让流程可以循环和分支。
LangGraph 内置 checkpointer，可以让同一个 thread_id 的对话保留历史。
interrupt() 可以让图暂停，等待人工审批后再用 Command(resume=...) 继续。

LangChain 是更高层的封装，提供模型接入、消息、工具、结构化输出等原语。
create_agent 就是用 LangGraph 实现的，所以它天然支持持久化和中断。
""".strip()


def main() -> None:
    # 1) 切分
    splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20)
    chunks = splitter.split_text(DOC)
    print(f"切分得到 {len(chunks)} 段：")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {c.replace(chr(10), ' ')}")
    print()

    # 2) 建库
    store = InMemoryVectorStore(embedding=HashEmbeddings(dim=64))
    store.add_texts(chunks)

    # 3) 检索
    question = "图怎么暂停等人工审批？"
    hits = store.similarity_search(question, k=2)
    print(f"问题：{question}")
    print("检索到的上下文：")
    context = "\n".join(f"- {d.page_content}" for d in hits)
    print(context)
    print()

    # 4) 拼提示词生成
    prompt = (
        "只根据下面的上下文回答问题，无法回答就说不知道。\n\n"
        f"上下文：\n{context}\n\n问题：{question}"
    )
    answer = get_model().invoke(prompt)
    print("回答：", answer.content)
    print(
        "\n提示：假 embedding 无真正语义，命中的可能是字面相近的段落。"
        "\n换成真实 embedding 后，同样的代码检索质量会显著提升。"
    )


if __name__ == "__main__":
    main()
