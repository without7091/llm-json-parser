# -*- coding: utf-8 -*-
"""
测试 LLMJsonParser 的核心功能与各清洗步骤。

涵盖：
  - 公开 API (parse / parse_safe / parse_all)
  - 阶段0：直接解析
  - 阶段1：Markdown 代码块提取
  - 阶段2：括号匹配定位
  - 清洗步骤1~5 逐步验证
  - 边界条件与异常处理
"""
from __future__ import absolute_import, unicode_literals, print_function

import json
import sys
import unittest

from llm_json_parser import LLMJsonParser, parse, parse_safe, parse_all


class TestDirectParse(unittest.TestCase):
    """阶段0：标准 JSON 直接解析（无需清洗）。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_simple_object(self):
        self.assertEqual(self.p.parse('{"a": 1}'), {"a": 1})

    def test_simple_array(self):
        self.assertEqual(self.p.parse('[1, 2, 3]'), [1, 2, 3])

    def test_nested(self):
        text = '{"a": {"b": [1, {"c": true}]}, "d": null}'
        result = self.p.parse(text)
        self.assertEqual(result, {"a": {"b": [1, {"c": True}]}, "d": None})

    def test_whitespace_around(self):
        self.assertEqual(self.p.parse('  \n  {"x": 1}  \n  '), {"x": 1})

    def test_empty_object(self):
        self.assertEqual(self.p.parse('{}'), {})

    def test_empty_array(self):
        self.assertEqual(self.p.parse('[]'), [])

    def test_string_value(self):
        self.assertEqual(self.p.parse('"hello"'), "hello")

    def test_number_value(self):
        self.assertEqual(self.p.parse('42'), 42)

    def test_boolean_and_null(self):
        self.assertEqual(self.p.parse('true'), True)
        self.assertEqual(self.p.parse('false'), False)
        self.assertIsNone(self.p.parse('null'))

    def test_unicode_content(self):
        text = '{"msg": "你好世界", "emoji": "🎉"}'
        result = self.p.parse(text)
        self.assertEqual(result["msg"], "你好世界")

    def test_bom_stripped(self):
        self.assertEqual(self.p.parse('\ufeff{"a": 1}'), {"a": 1})


class TestMarkdownExtraction(unittest.TestCase):
    """阶段1：从 Markdown 代码块中提取 JSON。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_json_code_block(self):
        text = 'blah\n```json\n{"a": 1}\n```\nblah'
        self.assertEqual(self.p.parse(text), {"a": 1})

    def test_no_language_tag(self):
        text = '```\n[1, 2]\n```'
        self.assertEqual(self.p.parse(text), [1, 2])

    def test_other_language_tag(self):
        text = '```python\n{"key": "value"}\n```'
        self.assertEqual(self.p.parse(text), {"key": "value"})

    def test_code_block_with_dirty_json(self):
        text = '```json\n{"a": True, "b": None,}\n```'
        self.assertEqual(self.p.parse(text), {"a": True, "b": None})

    def test_multiple_code_blocks_parse_returns_first(self):
        text = '```json\n{"first": 1}\n```\n```json\n{"second": 2}\n```'
        self.assertEqual(self.p.parse(text), {"first": 1})

    def test_multiple_code_blocks_parse_all(self):
        text = '```json\n{"a": 1}\n```\ntext\n```json\n{"b": 2}\n```'
        self.assertEqual(self.p.parse_all(text), [{"a": 1}, {"b": 2}])


class TestBracketMatching(unittest.TestCase):
    """阶段2：括号匹配从周围文本中定位 JSON。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_json_with_prefix(self):
        text = 'The result is: {"status": "ok"}'
        self.assertEqual(self.p.parse(text), {"status": "ok"})

    def test_json_with_suffix(self):
        text = '{"status": "ok"} — end of response.'
        self.assertEqual(self.p.parse(text), {"status": "ok"})

    def test_json_with_both(self):
        text = 'Here: {"a": 1} is the answer.'
        self.assertEqual(self.p.parse(text), {"a": 1})

    def test_array_in_text(self):
        text = 'Results: [1, 2, 3] done.'
        self.assertEqual(self.p.parse(text), [1, 2, 3])

    def test_brackets_inside_strings_not_confused(self):
        text = 'x {"msg": "a {b} [c]"} y'
        self.assertEqual(self.p.parse(text), {"msg": "a {b} [c]"})

    def test_multiple_json_in_text(self):
        text = 'A: {"x": 1} B: {"y": 2}'
        results = self.p.parse_all(text)
        self.assertEqual(results, [{"x": 1}, {"y": 2}])


class TestRemoveComments(unittest.TestCase):
    """清洗步骤1：移除 JS 风格注释。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_single_line_comment(self):
        text = '{"a": 1, // this is a comment\n"b": 2}'
        self.assertEqual(self.p.parse(text), {"a": 1, "b": 2})

    def test_multi_line_comment(self):
        text = '{"a": 1, /* multi\nline\ncomment */ "b": 2}'
        self.assertEqual(self.p.parse(text), {"a": 1, "b": 2})

    def test_comment_preserves_string_content(self):
        text = '{"url": "http://example.com"}'
        self.assertEqual(self.p.parse(text), {"url": "http://example.com"})

    def test_comment_at_end(self):
        text = '{"a": 1} // final comment'
        self.assertEqual(self.p.parse(text), {"a": 1})

    def test_slash_in_string_not_treated_as_comment(self):
        text = '{"path": "a/b/c", "regex": "x//y"}'
        self.assertEqual(
            self.p.parse(text),
            {"path": "a/b/c", "regex": "x//y"}
        )


class TestPythonLiterals(unittest.TestCase):
    """清洗步骤2：Python/JS 字面量替换。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_true_false_none(self):
        text = '{"a": True, "b": False, "c": None}'
        self.assertEqual(
            self.p.parse(text),
            {"a": True, "b": False, "c": None}
        )

    def test_undefined(self):
        text = '{"a": undefined}'
        self.assertEqual(self.p.parse(text), {"a": None})

    def test_no_replace_inside_string(self):
        text = '{"label": "True or False", "desc": "None of the above"}'
        result = self.p.parse(text)
        self.assertEqual(result["label"], "True or False")
        self.assertEqual(result["desc"], "None of the above")

    def test_no_replace_partial_word(self):
        text = '{"TrueValue": 1, "isFalse": 2, "NoneType": 3}'
        result = self.p.parse(text)
        # 这些键名不该被替换——但因为它们在字符串内，所以本来就不会被替换
        self.assertIn("TrueValue", result)

    def test_literal_at_boundaries(self):
        text = '[True,False,None]'
        self.assertEqual(self.p.parse(text), [True, False, None])


class TestTrailingCommas(unittest.TestCase):
    """清洗步骤3：去除尾随逗号。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_trailing_comma_object(self):
        text = '{"a": 1, "b": 2,}'
        self.assertEqual(self.p.parse(text), {"a": 1, "b": 2})

    def test_trailing_comma_array(self):
        text = '[1, 2, 3,]'
        self.assertEqual(self.p.parse(text), [1, 2, 3])

    def test_trailing_comma_nested(self):
        text = '{"a": [1, 2,], "b": {"c": 3,},}'
        self.assertEqual(self.p.parse(text), {"a": [1, 2], "b": {"c": 3}})

    def test_trailing_comma_with_whitespace(self):
        text = '{"a": 1 ,  \n  }'
        self.assertEqual(self.p.parse(text), {"a": 1})

    def test_comma_in_string_not_removed(self):
        text = '{"msg": "a, b, c,"}'
        self.assertEqual(self.p.parse(text), {"msg": "a, b, c,"})


class TestSingleQuotes(unittest.TestCase):
    """清洗步骤4：单引号转双引号。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_single_quoted_keys_and_values(self):
        text = "{'key': 'value'}"
        self.assertEqual(self.p.parse(text), {"key": "value"})

    def test_single_quoted_with_embedded_double_quote(self):
        text = """{'msg': 'he said "hello"'}"""
        result = self.p.parse(text)
        self.assertEqual(result, {"msg": 'he said "hello"'})

    def test_escaped_single_quote(self):
        text = "{'msg': 'it\\'s fine'}"
        result = self.p.parse(text)
        self.assertEqual(result, {"msg": "it's fine"})

    def test_mixed_quotes(self):
        text = """{'a': "double", "b": 'single'}"""
        result = self.p.parse(text)
        self.assertEqual(result, {"a": "double", "b": "single"})

    def test_no_single_quotes_passthrough(self):
        text = '{"a": "no single quotes here"}'
        self.assertEqual(self.p.parse(text), {"a": "no single quotes here"})


class TestUnescapedQuotesInValues(unittest.TestCase):
    """清洗步骤5：修复值字符串中未转义的引号。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_colon_pattern_in_value(self):
        """值中包含 ":"（用户报告的原始 bug）。"""
        text = '{"suggest": "软件管理":"应该写成xxx样子","rule":"软件管理不符合要求"}'
        result = self.p.parse(text)
        self.assertEqual(result["suggest"], '软件管理":"应该写成xxx样子')
        self.assertEqual(result["rule"], "软件管理不符合要求")

    def test_embedded_quotes_in_value(self):
        """值中包含引号包裹的词语。"""
        text = '{"msg": "say "hello" please", "ok": true}'
        result = self.p.parse(text)
        self.assertEqual(result["msg"], 'say "hello" please')
        self.assertTrue(result["ok"])

    def test_normal_json_unaffected(self):
        """正常 JSON 不受影响。"""
        text = '{"a": "hello", "b": 42, "c": true}'
        result = self.p.parse(text)
        self.assertEqual(result, {"a": "hello", "b": 42, "c": True})

    def test_nested_objects_unaffected(self):
        """嵌套对象不受影响。"""
        text = '{"a": {"b": "value"}, "c": [1, 2]}'
        result = self.p.parse(text)
        self.assertEqual(result, {"a": {"b": "value"}, "c": [1, 2]})

    def test_array_values_unaffected(self):
        """数组值不受影响。"""
        text = '["hello", "world", 42]'
        result = self.p.parse(text)
        self.assertEqual(result, ["hello", "world", 42])

    def test_multiple_embedded_quotes_in_one_value(self):
        """一个值中包含多个嵌入引号。"""
        text = '{"msg": "he said "hi" and she said "bye"", "done": true}'
        result = self.p.parse(text)
        self.assertEqual(result["msg"], 'he said "hi" and she said "bye"')
        self.assertTrue(result["done"])

    def test_empty_string_values_preserved(self):
        """空字符串值保持不变。"""
        text = '{"a": "", "b": "text"}'
        result = self.p.parse(text)
        self.assertEqual(result, {"a": "", "b": "text"})

    def test_already_escaped_quotes_not_double_escaped(self):
        """已转义的引号不被重复转义。"""
        text = '{"msg": "say \\"hello\\"", "ok": true}'
        result = self.p.parse(text)
        self.assertEqual(result["msg"], 'say "hello"')
        self.assertTrue(result["ok"])

    def test_embedded_quote_in_array_value(self):
        """数组中的值字符串包含嵌入引号。"""
        text = '["say "hello" please", "normal"]'
        result = self.p.parse(text)
        self.assertEqual(result[0], 'say "hello" please')
        self.assertEqual(result[1], "normal")

    def test_value_with_only_structural_after_close(self):
        """值字符串关闭引号后紧跟 } 的正常情况。"""
        text = '{"a": "value"}'
        result = self.p.parse(text)
        self.assertEqual(result, {"a": "value"})


class TestIncompleteJson(unittest.TestCase):
    """清洗步骤6：不完整 JSON 修复。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_missing_closing_brace(self):
        text = '{"a": 1, "b": 2'
        self.assertEqual(self.p.parse(text), {"a": 1, "b": 2})

    def test_missing_closing_bracket(self):
        text = '[1, 2, 3'
        self.assertEqual(self.p.parse(text), [1, 2, 3])

    def test_missing_nested_brackets(self):
        text = '{"a": [1, 2, {"b": 3'
        result = self.p.parse(text)
        self.assertEqual(result, {"a": [1, 2, {"b": 3}]})

    def test_unclosed_string(self):
        text = '{"name": "Alice'
        result = self.p.parse(text)
        self.assertEqual(result, {"name": "Alice"})

    def test_trailing_colon(self):
        text = '{"a": 1, "b":'
        result = self.p.parse(text)
        self.assertEqual(result, {"a": 1, "b": None})

    def test_trailing_comma_in_truncated(self):
        text = '{"a": 1,'
        result = self.p.parse(text)
        self.assertEqual(result, {"a": 1})

    def test_truncated_with_surrounding_text(self):
        text = 'Here is the result: {"name": "Alice", "items": [1, 2, 3'
        result = self.p.parse(text)
        self.assertEqual(result, {"name": "Alice", "items": [1, 2, 3]})


class TestCombinedIssues(unittest.TestCase):
    """多种问题组合出现。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_single_quotes_plus_python_literals(self):
        text = "{'active': True, 'count': None}"
        self.assertEqual(self.p.parse(text), {"active": True, "count": None})

    def test_comments_plus_trailing_commas(self):
        text = '{"a": 1, // comment\n"b": 2,}'
        self.assertEqual(self.p.parse(text), {"a": 1, "b": 2})

    def test_all_issues_combined(self):
        text = "{'a': True, 'tags': ['x', 'y',], /* note */ 'z': None}"
        result = self.p.parse(text)
        self.assertEqual(
            result,
            {"a": True, "tags": ["x", "y"], "z": None}
        )

    def test_markdown_with_dirty_content(self):
        text = """Sure, here you go:

```json
{
    "users": [
        {"id": 1, "name": "Alice", "active": True},
        {"id": 2, "name": "Bob",   "active": False},
    ],
    "total": 2,  // count
}
```

Hope this helps!
"""
        result = self.p.parse(text)
        self.assertEqual(result["total"], 2)
        self.assertEqual(len(result["users"]), 2)
        self.assertTrue(result["users"][0]["active"])


class TestErrorHandling(unittest.TestCase):
    """异常处理与边界条件。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_type_error_on_int(self):
        with self.assertRaises(TypeError):
            self.p.parse(123)

    def test_type_error_on_none(self):
        with self.assertRaises(TypeError):
            self.p.parse(None)

    def test_type_error_on_list(self):
        with self.assertRaises(TypeError):
            self.p.parse([1, 2])

    def test_value_error_on_garbage(self):
        with self.assertRaises(ValueError):
            self.p.parse("this is not json at all")

    def test_value_error_on_empty(self):
        with self.assertRaises(ValueError):
            self.p.parse("")

    def test_value_error_on_whitespace_only(self):
        with self.assertRaises(ValueError):
            self.p.parse("   \n\t  ")

    def test_parse_safe_returns_default(self):
        self.assertIsNone(self.p.parse_safe("garbage"))
        self.assertEqual(self.p.parse_safe("garbage", default={}), {})
        self.assertEqual(self.p.parse_safe("garbage", default=42), 42)

    def test_parse_safe_returns_result_on_success(self):
        self.assertEqual(self.p.parse_safe('{"a": 1}'), {"a": 1})

    def test_parse_safe_type_error_returns_default(self):
        self.assertEqual(self.p.parse_safe(123, default="fallback"), "fallback")


class TestParseAll(unittest.TestCase):
    """parse_all 方法测试。"""

    def setUp(self):
        self.p = LLMJsonParser()

    def test_no_json(self):
        self.assertEqual(self.p.parse_all("no json here"), [])

    def test_single_json(self):
        self.assertEqual(self.p.parse_all('{"a": 1}'), [{"a": 1}])

    def test_multiple_objects(self):
        text = '{"a": 1} and {"b": 2} and {"c": 3}'
        self.assertEqual(
            self.p.parse_all(text),
            [{"a": 1}, {"b": 2}, {"c": 3}]
        )

    def test_mixed_objects_and_arrays(self):
        text = '{"x": 1} then [1, 2]'
        self.assertEqual(self.p.parse_all(text), [{"x": 1}, [1, 2]])

    def test_markdown_blocks_preferred(self):
        text = '```json\n{"a": 1}\n```\n{"b": 2}'
        # Markdown block takes priority; bare JSON outside is ignored
        self.assertEqual(self.p.parse_all(text), [{"a": 1}])


class TestModuleLevelFunctions(unittest.TestCase):
    """模块级便捷函数。"""

    def test_parse(self):
        self.assertEqual(parse('{"a": 1}'), {"a": 1})

    def test_parse_safe(self):
        self.assertIsNone(parse_safe("nope"))
        self.assertEqual(parse_safe("nope", default=[]), [])

    def test_parse_all(self):
        self.assertEqual(parse_all('[1] [2]'), [[1], [2]])


if __name__ == '__main__':
    unittest.main()
