"""第 3 课：结构化输出 —— 用 Pydantic 约束模型返回类型。

核心认知：
    with_structured_output 内部会转成 tool/JSON schema 调用，
    返回的是**已验证的 Pydantic 对象**，不是字符串，不用自己 json.loads。

运行：uv run python -m core.l3_structured
"""

from pydantic import BaseModel, Field

from common.config import get_model


class Person(BaseModel):
    """从文本中抽取的人物信息。"""

    name: str = Field(description="姓名")
    age: int = Field(description="年龄；文中未提及则填 -1")
    skills: list[str] = Field(default_factory=list, description="掌握的技能")
    seniority: str = Field(
        default="unknown",
        description="资历：junior / mid / senior / unknown",
    )


def main() -> None:
    model = get_model()
    extractor = model.with_structured_output(Person)

    text = "张伟今年 32 岁，做了 8 年后端，会 Python、Kubernetes，也会一点 Rust。"
    person = extractor.invoke(f"抽取人物信息：{text}")

    print("返回类型：", type(person).__name__)
    print(person.model_dump_json(indent=2))
    print("\n直接取字段：", person.name, person.age, person.seniority)


if __name__ == "__main__":
    main()
