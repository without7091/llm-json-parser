# -*- coding: utf-8 -*-
"""
模拟真实大模型（GPT / Claude / Gemini / 通义千问 / 文心一言等）
返回异常 JSON 的典型场景进行测试。

每个测试用例的输入都是从实际大模型交互中总结出来的高频错误模式。
"""
from __future__ import absolute_import, unicode_literals, print_function

import unittest

from llm_json_parser import LLMJsonParser


class TestGPTStyleOutput(unittest.TestCase):
    """GPT 系列常见的输出格式。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_chatgpt_markdown_with_explanation(self):
        """ChatGPT 经常在代码块前后加解释文字。"""
        text = """Sure! Here's the JSON you requested:

```json
{
  "name": "ChatGPT",
  "version": "4",
  "capabilities": ["text", "code", "analysis"],
  "multilingual": true
}
```

Let me know if you need anything else!"""
        result = self.p.parse(text)
        self.assertEqual(result["name"], "ChatGPT")
        self.assertEqual(result["capabilities"], ["text", "code", "analysis"])

    def test_chatgpt_truncated_long_response(self):
        """ChatGPT 输出过长时会被截断（token limit）。"""
        text = """{
  "users": [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
    {"id": 3, "name": "Charlie", "email": "charlie@exa"""
        result = self.p.parse(text)
        # 截断修复后应得到包含 users 数组的外层对象
        self.assertIn("users", result)
        self.assertIsInstance(result["users"], list)
        self.assertGreaterEqual(len(result["users"]), 2)

    def test_chatgpt_python_dict_format(self):
        """用户要求 JSON 但 GPT 返回了 Python dict 格式。"""
        text = """Here is the data:

{'students': [
    {'name': 'Alice', 'grade': 'A', 'passed': True},
    {'name': 'Bob', 'grade': 'B', 'passed': True},
    {'name': 'Charlie', 'grade': 'F', 'passed': False}
], 'count': 3, 'has_failures': True}"""
        result = self.p.parse(text)
        self.assertEqual(result["count"], 3)
        self.assertTrue(result["students"][0]["passed"])
        self.assertFalse(result["students"][2]["passed"])

    def test_chatgpt_with_line_comments(self):
        """GPT 有时在 JSON 中加 // 注释说明每个字段。"""
        text = """{
  "model": "gpt-4",       // the model name
  "temperature": 0.7,     // creativity level
  "max_tokens": 2048,     // maximum output length
  "stream": true          // enable streaming
}"""
        result = self.p.parse(text)
        self.assertEqual(result["model"], "gpt-4")
        self.assertEqual(result["temperature"], 0.7)
        self.assertTrue(result["stream"])


class TestClaudeStyleOutput(unittest.TestCase):
    """Claude 系列常见输出格式。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_claude_structured_response(self):
        """Claude 倾向于输出结构化的回答，JSON 前有分析文字。"""
        text = """Based on your requirements, I've structured the configuration as follows:

```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "myapp_db",
    "pool_size": 10
  },
  "cache": {
    "enabled": true,
    "ttl": 3600,
    "backend": "redis"
  },
  "logging": {
    "level": "INFO",
    "format": "json"
  }
}
```

This configuration includes sensible defaults for a production environment."""
        result = self.p.parse(text)
        self.assertEqual(result["database"]["port"], 5432)
        self.assertTrue(result["cache"]["enabled"])
        self.assertEqual(result["logging"]["level"], "INFO")

    def test_claude_multiple_json_blocks(self):
        """Claude 有时在一个回答中给出请求和响应两个 JSON。"""
        text = """Here's the request:

```json
{"method": "POST", "url": "/api/users", "body": {"name": "test"}}
```

And the expected response:

```json
{"status": 201, "data": {"id": 1, "name": "test"}}
```
"""
        results = self.p.parse_all(text)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["method"], "POST")
        self.assertEqual(results[1]["status"], 201)


class TestGeminiStyleOutput(unittest.TestCase):
    """Gemini 常见输出格式。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_gemini_mixed_literals(self):
        """Gemini 偶尔混用 Python 和 JSON 字面量。"""
        text = """{
    "results": [
        {"query": "test", "found": True, "count": 42},
        {"query": "empty", "found": False, "count": 0}
    ],
    "metadata": {
        "source": "gemini",
        "cached": None,
        "version": undefined
    }
}"""
        result = self.p.parse(text)
        self.assertTrue(result["results"][0]["found"])
        self.assertFalse(result["results"][1]["found"])
        self.assertIsNone(result["metadata"]["cached"])
        self.assertIsNone(result["metadata"]["version"])  # undefined → null

    def test_gemini_trailing_commas_everywhere(self):
        """Gemini 返回的 JSON 经常有尾随逗号。"""
        text = """{
    "items": [
        "apple",
        "banana",
        "cherry",
    ],
    "nested": {
        "a": 1,
        "b": 2,
    },
}"""
        result = self.p.parse(text)
        self.assertEqual(result["items"], ["apple", "banana", "cherry"])
        self.assertEqual(result["nested"]["b"], 2)


class TestChineseLLMOutput(unittest.TestCase):
    """中文大模型（通义千问、文心一言、智谱等）常见输出格式。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_qianwen_chinese_explanation(self):
        """通义千问习惯在 JSON 前后加中文解释。"""
        text = """根据您的要求，我已整理出以下JSON格式的数据：

```json
{
    "城市": "上海",
    "人口": 24870000,
    "GDP": "4.32万亿",
    "别名": ["魔都", "东方巴黎", "沪"]
}
```

以上数据来源于2023年统计年鉴。如需更多信息请告知。"""
        result = self.p.parse(text)
        self.assertEqual(result["城市"], "上海")
        self.assertEqual(len(result["别名"]), 3)

    def test_wenxin_single_quotes_chinese(self):
        """文心一言有时用单引号输出 JSON。"""
        text = "{'姓名': '张三', '年龄': 28, '已婚': False, '住址': None}"
        result = self.p.parse(text)
        self.assertEqual(result["姓名"], "张三")
        self.assertEqual(result["年龄"], 28)
        self.assertFalse(result["已婚"])
        self.assertIsNone(result["住址"])

    def test_zhipu_block_comment_style(self):
        """智谱GLM偶尔在JSON中加注释。"""
        text = """{
    "任务": "文本分类",
    /* 以下为分类结果 */
    "结果": [
        {"文本": "今天天气真好", "类别": "日常", "置信度": 0.95},
        {"文本": "股票大涨", "类别": "财经", "置信度": 0.88}
    ],
    "模型": "GLM-4" // 使用的模型版本
}"""
        result = self.p.parse(text)
        self.assertEqual(result["任务"], "文本分类")
        self.assertEqual(len(result["结果"]), 2)
        self.assertEqual(result["模型"], "GLM-4")

    def test_chinese_llm_no_markdown(self):
        """中文模型直接输出 JSON 不加代码块，前后有文字。"""
        text = '分析结果如下：{"score": 85, "level": "优秀", "pass": true}，建议继续保持。'
        result = self.p.parse(text)
        self.assertEqual(result["score"], 85)
        self.assertEqual(result["level"], "优秀")


class TestStreamingTruncation(unittest.TestCase):
    """流式输出中途断开导致的截断。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_truncated_mid_string(self):
        """字符串值写到一半被截断。"""
        text = '{"title": "Introduction to Machine Lear'
        result = self.p.parse(text)
        self.assertIn("title", result)

    def test_truncated_mid_array(self):
        """数组写到一半被截断。"""
        text = '{"tags": ["python", "json", "parsing'
        result = self.p.parse(text)
        self.assertIn("tags", result)

    def test_truncated_after_key(self):
        """写完键名和冒号后被截断。"""
        text = '{"name": "Alice", "age":'
        result = self.p.parse(text)
        self.assertEqual(result["name"], "Alice")
        self.assertIsNone(result["age"])

    def test_truncated_after_comma(self):
        """写完一个条目和逗号后被截断。"""
        text = '{"a": 1, "b": 2,'
        result = self.p.parse(text)
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_truncated_deep_nesting(self):
        """深层嵌套结构截断。"""
        text = '{"a": {"b": {"c": [1, 2, {"d": "val'
        result = self.p.parse(text)
        self.assertIn("a", result)

    def test_truncated_with_prefix_text(self):
        """有前缀文字的截断 JSON。"""
        text = 'The API returned: {"status": "ok", "data": [1, 2'
        result = self.p.parse(text)
        self.assertEqual(result["status"], "ok")


class TestComplexRealWorldScenarios(unittest.TestCase):
    """复杂的真实世界场景。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_api_response_in_explanation(self):
        """LLM 在解释中嵌入了 API 响应示例。"""
        text = """The API will return a response like this:

{"status": 200, "message": "Success", "data": {"user_id": 123, "token": "abc-def-ghi"}}

You can use the token for subsequent requests."""
        result = self.p.parse(text)
        self.assertEqual(result["status"], 200)
        self.assertEqual(result["data"]["token"], "abc-def-ghi")

    def test_json_with_escaped_quotes_in_values(self):
        """值中包含转义引号。"""
        text = '{"code": "print(\\"hello\\")", "lang": "python"}'
        result = self.p.parse(text)
        self.assertEqual(result["code"], 'print("hello")')

    def test_json_with_newlines_in_code_block(self):
        """代码块中 JSON 带有很多空行。"""
        text = """```json

{

    "key": "value",

    "number": 42

}

```"""
        result = self.p.parse(text)
        self.assertEqual(result["key"], "value")

    def test_large_nested_structure(self):
        """较大的嵌套结构。"""
        text = """{
    "company": "Acme Corp",
    "departments": [
        {
            "name": "Engineering",
            "teams": [
                {"name": "Backend", "members": 15, "lead": "Alice"},
                {"name": "Frontend", "members": 10, "lead": "Bob"},
                {"name": "DevOps", "members": 5, "lead": "Charlie"}
            ]
        },
        {
            "name": "Marketing",
            "teams": [
                {"name": "Growth", "members": 8, "lead": "Diana"},
                {"name": "Content", "members": 6, "lead": "Eve"}
            ]
        }
    ],
    "total_employees": 44
}"""
        result = self.p.parse(text)
        self.assertEqual(result["company"], "Acme Corp")
        self.assertEqual(len(result["departments"]), 2)
        self.assertEqual(result["total_employees"], 44)

    def test_function_call_json_output(self):
        """LLM function calling 场景的 JSON 输出。"""
        text = """I'll call the search function:

```json
{
    "function": "search_web",
    "parameters": {
        "query": "Python JSON parser",
        "max_results": 5,
        "language": "zh-CN",
        "safe_search": true
    }
}
```"""
        result = self.p.parse(text)
        self.assertEqual(result["function"], "search_web")
        self.assertEqual(result["parameters"]["max_results"], 5)

    def test_agent_tool_use_format(self):
        """Agent 工具调用格式（混合问题）。"""
        text = """{'action': 'search', 'action_input': {'query': 'weather today', 'location': 'Beijing'}, 'thought': "I need to check today's weather"}"""
        result = self.p.parse(text)
        self.assertEqual(result["action"], "search")
        self.assertEqual(result["action_input"]["location"], "Beijing")

    def test_jsonl_first_line(self):
        """JSONL 格式（每行一个 JSON），parse 取第一个。"""
        text = '{"id": 1, "text": "first"}\n{"id": 2, "text": "second"}\n{"id": 3, "text": "third"}'
        result = self.p.parse(text)
        self.assertEqual(result["id"], 1)

    def test_jsonl_all_lines(self):
        """JSONL 格式，parse_all 取全部。"""
        text = '{"id": 1}\n{"id": 2}\n{"id": 3}'
        results = self.p.parse_all(text)
        self.assertEqual(len(results), 3)
        self.assertEqual([r["id"] for r in results], [1, 2, 3])

    def test_json_with_url_containing_slashes(self):
        """JSON 中包含 URL（含 //），不应被当作注释。"""
        text = '{"homepage": "https://example.com/path", "api": "http://api.example.com"}'
        result = self.p.parse(text)
        self.assertEqual(result["homepage"], "https://example.com/path")
        self.assertEqual(result["api"], "http://api.example.com")

    def test_mixed_array_of_objects(self):
        """LLM 返回的结构化列表结果。"""
        text = """Here are the results:

[
    {"title": "Result 1", "score": 0.95, "relevant": True},
    {"title": "Result 2", "score": 0.82, "relevant": True},
    {"title": "Result 3", "score": 0.45, "relevant": False}
]"""
        result = self.p.parse(text)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0]["score"], 0.95)


class TestUnescapedQuotesInLLMOutput(unittest.TestCase):
    """LLM 输出中值字符串包含未转义引号的场景。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_chinese_llm_suggest_with_colon_quote_pattern(self):
        """中文 LLM 返回含 ":" 模式的建议字段（用户报告的原始 bug）。"""
        text = '{"suggest": "软件管理":"应该写成xxx样子","rule":"软件管理不符合要求"}'
        result = self.p.parse(text)
        self.assertEqual(result["suggest"], '软件管理":"应该写成xxx样子')
        self.assertEqual(result["rule"], "软件管理不符合要求")

    def test_llm_quoting_dialogue_in_value(self):
        """LLM 在值中引用对话内容。"""
        text = '{"analysis": "用户说 "我不同意" 然后离开了", "sentiment": "negative"}'
        result = self.p.parse(text)
        self.assertEqual(result["analysis"], '用户说 "我不同意" 然后离开了')
        self.assertEqual(result["sentiment"], "negative")

    def test_llm_code_snippet_in_value(self):
        """LLM 在值中包含代码片段，含引号。"""
        text = '{"fix": "将 "name" 改为 "username"", "file": "config.py"}'
        result = self.p.parse(text)
        self.assertEqual(result["fix"], '将 "name" 改为 "username"')
        self.assertEqual(result["file"], "config.py")

    def test_llm_multiple_fields_with_embedded_quotes(self):
        """多个字段都有嵌入引号。"""
        text = '{"error": "字段 "age" 不合法", "fix": "将 "age" 设为整数", "status": "failed"}'
        result = self.p.parse(text)
        self.assertEqual(result["error"], '字段 "age" 不合法')
        self.assertEqual(result["fix"], '将 "age" 设为整数')
        self.assertEqual(result["status"], "failed")

    def test_llm_output_with_markdown_and_embedded_quotes(self):
        """Markdown 代码块中的 JSON 也包含未转义引号。"""
        text = '```json\n{"msg": "请使用 "utf-8" 编码", "ok": true}\n```'
        result = self.p.parse(text)
        self.assertEqual(result["msg"], '请使用 "utf-8" 编码')
        self.assertTrue(result["ok"])


class TestKnownLimitations(unittest.TestCase):
    """已知的局限性——这些用例预期会失败或行为不确定。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_unquoted_keys_fails(self):
        """无引号键名不受支持。"""
        text = '{name: "Alice", age: 30}'
        result = self.p.parse_safe(text)
        # 这个不一定能解析成功，取决于具体实现
        # 主要验证不会抛出未预期的异常
        # (parse_safe 会返回 None 如果解析失败)
        self.assertTrue(result is None or isinstance(result, dict))

    def test_nan_infinity_not_supported(self):
        """NaN 和 Infinity — CPython 的 json.loads 实际上接受它们（非标准），
        所以这里验证解析器不会崩溃，而非断言失败。"""
        text = '{"value": NaN}'
        result = self.p.parse_safe(text)
        # CPython 接受 NaN，结果可能为 dict 也可能为 None（取决于实现）
        # 主要确保不抛出未预期的异常
        self.assertTrue(result is None or isinstance(result, dict))

    def test_deeply_truncated_key(self):
        """键名中间截断——无法推断完整键名。"""
        text = '{"us'
        result = self.p.parse_safe(text)
        # 可能解析出 {"us": ...} 或失败，均可接受
        self.assertTrue(result is None or isinstance(result, dict))


if __name__ == '__main__':
    unittest.main()
