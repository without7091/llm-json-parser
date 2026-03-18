# -*- coding: utf-8 -*-
"""
LLMJsonParser - 从大模型输出中提取并修复 JSON 的工具类。

兼容性：Python 2.7+ 和 Python 3.x 全系列
依赖：仅使用标准库（json, re, sys）
"""
from __future__ import absolute_import, unicode_literals, print_function

import json
import re
import sys

PY2 = sys.version_info[0] == 2
if PY2:
    string_types = (str, unicode)  # noqa: F821
else:
    string_types = (str,)


class LLMJsonParser(object):
    """
    从大模型（LLM）回答中健壮地提取和修复 JSON。

    支持：
      - Markdown 代码块包裹（```json ... ```）
      - JSON 前后有额外文本
      - 尾随逗号（trailing commas）
      - Python 风格字面量（True/False/None → true/false/null）
      - JavaScript 注释（// 和 /* */）
      - 单引号代替双引号（含撇号的字符串）
      - 截断/不完整的 JSON（补全括号）
      - 多个 JSON 块（parse_all）

    用法::

        parser = LLMJsonParser()

        # 提取第一个有效 JSON
        result = parser.parse(llm_output)

        # 安全解析（失败返回默认值）
        result = parser.parse_safe(llm_output, default={})

        # 提取所有 JSON
        results = parser.parse_all(llm_output)
    """

    _SENTINEL = object()

    _RE_MARKDOWN_BLOCK = re.compile(
        r'```(?:[a-zA-Z]*)?\s*\n?([\s\S]*?)\n?\s*```',
        re.DOTALL
    )

    _LITERAL_REPLACEMENTS = [
        ('undefined', 'null'),
        ('False', 'false'),
        ('True', 'true'),
        ('None', 'null'),
    ]

    _CLOSING = {'{': '}', '[': ']'}
    _OPENING = set('{[')
    _CLOSING_CHARS = set('}]')

    # ── 公开 API ──────────────────────────────────────────────────────────────

    def parse(self, text):
        """
        从文本中提取并解析第一个有效 JSON 对象或数组。

        :param text: 包含 JSON 的字符串（可能有额外文本、格式问题等）
        :returns: 解析后的 Python 对象（dict/list/str/int/float/bool/None）
        :raises TypeError: 输入不是字符串
        :raises ValueError: 无法从文本中提取有效 JSON
        """
        text = self._ensure_text(text)
        result = self._parse_internal(text)
        if result is self._SENTINEL:
            raise ValueError(
                "Could not extract valid JSON from the provided text "
                "(tried {0} characters)".format(len(text))
            )
        return result

    def parse_safe(self, text, default=None):
        """
        从文本中提取并解析 JSON，失败时返回默认值而非抛出异常。

        :param text: 包含 JSON 的字符串
        :param default: 解析失败时的返回值（默认 None）
        :returns: 解析结果或 default
        """
        try:
            return self.parse(text)
        except (TypeError, ValueError):
            return default

    def parse_all(self, text):
        """
        从文本中提取所有 JSON 对象/数组。

        :param text: 可能包含多个 JSON 块的字符串
        :returns: list，包含所有成功解析的 Python 对象
        """
        text = self._ensure_text(text)
        return self._parse_all_internal(text)

    # ── 内部主流程 ────────────────────────────────────────────────────────────

    def _ensure_text(self, text):
        if not isinstance(text, string_types):
            raise TypeError(
                "Expected str, got {0}".format(type(text).__name__)
            )
        if PY2 and isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        if text.startswith('\ufeff'):
            text = text[1:]
        return text

    def _parse_internal(self, text):
        text = text.strip()
        if not text:
            return self._SENTINEL

        # 阶段0：快速路径——直接解析
        result = self._try_loads(text)
        if result is not self._SENTINEL:
            return result

        # 阶段1：从 Markdown 代码块提取
        for candidate in self._extract_from_markdown(text):
            result = self._try_clean_and_parse(candidate)
            if result is not self._SENTINEL:
                return result

        # 阶段2+2.5：括号匹配定位 JSON + 截断修复
        #
        # 策略：收集所有字符串外的开括号位置。对每个位置：
        #   - 先尝试完整匹配（_find_matching_end）提取精确边界
        #   - 匹配失败时（截断），取该位置到文本末尾尝试清洗+修复
        # 按位置顺序处理，确保最外层优先于内层。
        boundaries = self._find_json_boundaries(text)
        boundary_map = {}  # start → end for complete matches
        for start, end in boundaries:
            boundary_map[start] = end

        all_opening_positions = []
        _sc_in_str = False
        _sc_qc = None
        _sc_skip = False
        for idx in range(len(text)):
            if _sc_skip:
                _sc_skip = False
                continue
            ch = text[idx]
            if _sc_in_str:
                if ch == '\\':
                    _sc_skip = True
                    continue
                if ch == _sc_qc:
                    _sc_in_str = False
            else:
                if ch in ('"', "'"):
                    _sc_in_str = True
                    _sc_qc = ch
                elif ch in self._OPENING:
                    all_opening_positions.append(idx)

        skip_until = -1
        for pos in all_opening_positions:
            if pos < skip_until:
                continue
            if pos in boundary_map:
                # 完整匹配：用精确边界
                candidate = text[pos:boundary_map[pos]]
                end_pos = boundary_map[pos]
            else:
                # 截断：从此位置到文本末尾
                candidate = text[pos:]
                end_pos = len(text)
            result = self._try_clean_and_parse(candidate)
            if result is not self._SENTINEL:
                return result
            # 如果完整匹配失败，跳过其内部的开括号
            if pos in boundary_map:
                skip_until = boundary_map[pos]

        # 阶段3：对整个文本应用清洗
        result = self._try_clean_and_parse(text)
        if result is not self._SENTINEL:
            return result

        return self._SENTINEL

    def _parse_all_internal(self, text):
        results = []

        # 优先处理 Markdown 块
        markdown_candidates = self._extract_from_markdown(text)
        if markdown_candidates:
            for candidate in markdown_candidates:
                result = self._try_clean_and_parse(candidate)
                if result is not self._SENTINEL:
                    results.append(result)
            if results:
                return results

        # 扫描所有顶层 JSON 边界
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch in self._OPENING:
                end = self._find_matching_end(text, i)
                if end is not None:
                    candidate = text[i:end]
                    result = self._try_clean_and_parse(candidate)
                    if result is not self._SENTINEL:
                        results.append(result)
                    i = end
                    continue
            i += 1

        return results

    # ── 尝试解析 ──────────────────────────────────────────────────────────────

    def _try_loads(self, text):
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return self._SENTINEL

    def _try_clean_and_parse(self, text):
        """渐进式清洗并尝试解析，每步后立即尝试，成功即返回。"""
        text = text.strip()
        if not text:
            return self._SENTINEL

        # Step 0：原文直接解析
        result = self._try_loads(text)
        if result is not self._SENTINEL:
            return result

        # Step 1：去除 JS 注释
        s1 = self._remove_comments(text)
        result = self._try_loads(s1)
        if result is not self._SENTINEL:
            return result

        # Step 2：替换 Python/JS 字面量
        s2 = self._fix_python_literals(s1)
        result = self._try_loads(s2)
        if result is not self._SENTINEL:
            return result

        # Step 3：去除尾随逗号
        s3 = self._fix_trailing_commas(s2)
        result = self._try_loads(s3)
        if result is not self._SENTINEL:
            return result

        # Step 4：单引号 → 双引号
        s4 = self._fix_single_quotes(s3)
        result = self._try_loads(s4)
        if result is not self._SENTINEL:
            return result

        # Step 5：修复值字符串中未转义的引号
        s5 = self._fix_unescaped_quotes_in_values(s4)
        result = self._try_loads(s5)
        if result is not self._SENTINEL:
            return result

        # Step 6：不完整 JSON 修复（最后手段）
        s6 = self._fix_incomplete_json(s5)
        result = self._try_loads(s6)
        if result is not self._SENTINEL:
            return result

        return self._SENTINEL

    # ── Markdown 提取 ─────────────────────────────────────────────────────────

    def _extract_from_markdown(self, text):
        results = []
        for match in self._RE_MARKDOWN_BLOCK.finditer(text):
            content = match.group(1).strip()
            if content:
                results.append(content)
        return results

    # ── 括号匹配 ──────────────────────────────────────────────────────────────

    def _find_json_boundaries(self, text):
        """扫描文本，找出所有可能的 JSON 对象/数组边界，返回 [(start, end), ...]。"""
        boundaries = []
        i = 0
        n = len(text)
        in_string = False
        quote_char = None

        while i < n:
            ch = text[i]
            if in_string:
                if ch == '\\':
                    i += 2
                    continue
                if ch == quote_char:
                    in_string = False
            else:
                if ch in ('"', "'"):
                    in_string = True
                    quote_char = ch
                elif ch in self._OPENING:
                    end = self._find_matching_end(text, i)
                    if end is not None:
                        boundaries.append((i, end))
                        i = end
                        continue
            i += 1

        return boundaries

    def _find_matching_end(self, text, start):
        """从 start 开始，找到匹配的 JSON 结构结束位置（不含），返回位置或 None。"""
        n = len(text)
        stack = []
        in_string = False
        quote_char = None
        i = start

        while i < n:
            ch = text[i]
            if in_string:
                if ch == '\\':
                    i += 2
                    continue
                if ch == quote_char:
                    in_string = False
                i += 1
                continue

            if ch in ('"', "'"):
                in_string = True
                quote_char = ch
            elif ch in self._OPENING:
                stack.append(ch)
            elif ch == '}':
                if not stack or stack[-1] != '{':
                    return None
                stack.pop()
                if not stack:
                    return i + 1
            elif ch == ']':
                if not stack or stack[-1] != '[':
                    return None
                stack.pop()
                if not stack:
                    return i + 1

            i += 1

        return None

    # ── 清洗步骤 1：去除 JS 注释 ─────────────────────────────────────────────

    def _remove_comments(self, text):
        """移除 // 单行注释和 /* */ 多行注释，保留字符串内的注释标记。"""
        result = []
        i = 0
        n = len(text)
        in_string = False
        quote_char = None

        while i < n:
            ch = text[i]

            if in_string:
                result.append(ch)
                if ch == '\\' and i + 1 < n:
                    result.append(text[i + 1])
                    i += 2
                    continue
                if ch == quote_char:
                    in_string = False
                i += 1
                continue

            if ch in ('"', "'"):
                in_string = True
                quote_char = ch
                result.append(ch)
                i += 1
            elif ch == '/' and i + 1 < n:
                next_ch = text[i + 1]
                if next_ch == '/':
                    # 单行注释：跳过直到行尾
                    i += 2
                    while i < n and text[i] != '\n':
                        i += 1
                elif next_ch == '*':
                    # 多行注释：跳过直到 */
                    i += 2
                    while i < n:
                        if text[i] == '*' and i + 1 < n and text[i + 1] == '/':
                            i += 2
                            break
                        i += 1
                else:
                    result.append(ch)
                    i += 1
            else:
                result.append(ch)
                i += 1

        return ''.join(result)

    # ── 清洗步骤 2：Python/JS 字面量替换 ────────────────────────────────────

    def _fix_python_literals(self, text):
        """
        在字符串外部，将 Python/JS 字面量替换为 JSON 等价物。
        使用单词边界检查，避免替换 TrueValue 等包含目标词的标识符。
        """
        result = []
        i = 0
        n = len(text)
        in_string = False
        quote_char = None

        while i < n:
            ch = text[i]

            if in_string:
                result.append(ch)
                if ch == '\\' and i + 1 < n:
                    result.append(text[i + 1])
                    i += 2
                    continue
                if ch == quote_char:
                    in_string = False
                i += 1
                continue

            if ch in ('"', "'"):
                in_string = True
                quote_char = ch
                result.append(ch)
                i += 1
                continue

            replaced = False
            for old, new in self._LITERAL_REPLACEMENTS:
                old_len = len(old)
                if text[i:i + old_len] != old:
                    continue
                pre_ok = (
                    i == 0
                    or not (text[i - 1].isalpha() or text[i - 1] == '_')
                )
                post_pos = i + old_len
                post_ok = (
                    post_pos >= n
                    or not (
                        text[post_pos].isalpha()
                        or text[post_pos] == '_'
                        or text[post_pos].isdigit()
                    )
                )
                if pre_ok and post_ok:
                    result.append(new)
                    i += old_len
                    replaced = True
                    break

            if not replaced:
                result.append(ch)
                i += 1

        return ''.join(result)

    # ── 清洗步骤 3：去除尾随逗号 ────────────────────────────────────────────

    def _fix_trailing_commas(self, text):
        """移除 JSON 中的尾随逗号（位于 } 或 ] 之前的逗号）。"""
        result = []
        i = 0
        n = len(text)
        in_string = False
        quote_char = None

        while i < n:
            ch = text[i]

            if in_string:
                result.append(ch)
                if ch == '\\' and i + 1 < n:
                    result.append(text[i + 1])
                    i += 2
                    continue
                if ch == quote_char:
                    in_string = False
                i += 1
                continue

            if ch in ('"', "'"):
                in_string = True
                quote_char = ch
                result.append(ch)
                i += 1
                continue

            if ch == ',':
                j = i + 1
                while j < n and text[j] in ' \t\n\r':
                    j += 1
                if j < n and text[j] in ('}', ']'):
                    # 尾随逗号：跳过
                    i += 1
                    continue

            result.append(ch)
            i += 1

        return ''.join(result)

    # ── 清洗步骤 4：单引号 → 双引号 ─────────────────────────────────────────

    def _fix_single_quotes(self, text):
        """
        将单引号定界的字符串转换为双引号定界。

        规则：
          - 单引号字符串内的 \\' → 输出字面量 '（双引号字符串内无需转义单引号）
          - 单引号字符串内的裸 " → 输出 \\"（双引号需转义）
          - 双引号字符串内的内容原样输出
        """
        if "'" not in text:
            return text

        result = []
        i = 0
        n = len(text)
        in_string = False
        quote_char = None

        while i < n:
            ch = text[i]

            if not in_string:
                if ch == "'":
                    in_string = True
                    quote_char = "'"
                    result.append('"')
                    i += 1
                elif ch == '"':
                    in_string = True
                    quote_char = '"'
                    result.append('"')
                    i += 1
                else:
                    result.append(ch)
                    i += 1
                continue

            # 在字符串内
            if ch == '\\' and i + 1 < n:
                next_ch = text[i + 1]
                if quote_char == "'":
                    if next_ch == "'":
                        # \' 在单引号字符串中 = 字面量单引号，双引号字符串内无需转义
                        result.append("'")
                        i += 2
                    elif next_ch == '"':
                        # \" 在双引号字符串中仍需转义
                        result.append('\\"')
                        i += 2
                    else:
                        result.append(ch)
                        result.append(next_ch)
                        i += 2
                else:
                    # 双引号字符串内的转义序列原样保留
                    result.append(ch)
                    result.append(next_ch)
                    i += 2

            elif ch == quote_char:
                in_string = False
                result.append('"')
                i += 1

            elif ch == '"' and quote_char == "'":
                # 单引号字符串内出现裸双引号 → 转义
                result.append('\\"')
                i += 1

            else:
                result.append(ch)
                i += 1

        return ''.join(result)

    # ── 清洗步骤 5：修复值字符串中未转义的引号 ──────────────────────────────

    def _fix_unescaped_quotes_in_values(self, text):
        """
        修复 JSON 值字符串中未转义的双引号。

        核心启发式：在合法 JSON 中，值字符串的关闭引号后面（忽略空白）
        必须是 , } ] 或输入结束。如果一个引号后面跟的是其他字符，
        说明它是嵌入引号，应转义为 \\"。

        此启发式不适用于键字符串，因为 "key": 是正常语法。
        """
        n = len(text)
        if n == 0:
            return text

        result = []
        i = 0
        # State tracking
        stack = []          # 'object' or 'array'
        expect_value = False
        in_string = False
        is_value_string = False

        while i < n:
            ch = text[i]

            if in_string:
                # Handle escape sequences
                if ch == '\\' and i + 1 < n:
                    result.append(ch)
                    result.append(text[i + 1])
                    i += 2
                    continue

                if ch == '"':
                    if not is_value_string:
                        # Key string — closing quote is always real
                        result.append(ch)
                        in_string = False
                        i += 1
                        continue

                    # Value string — look ahead to decide if this is the real close
                    j = i + 1
                    while j < n and text[j] in ' \t\n\r':
                        j += 1

                    if j >= n or text[j] in (',', '}', ']'):
                        # Real closing quote
                        result.append(ch)
                        in_string = False
                        i += 1
                        continue
                    else:
                        # Embedded quote — escape it
                        result.append('\\')
                        result.append('"')
                        i += 1
                        continue

                # Normal character inside string
                result.append(ch)
                i += 1
                continue

            # Outside any string
            if ch == '"':
                in_string = True
                is_value_string = expect_value
                result.append(ch)
                i += 1
                continue

            if ch == '{':
                stack.append('object')
                expect_value = False  # next token should be key (or })
                result.append(ch)
                i += 1
                continue

            if ch == '[':
                stack.append('array')
                expect_value = True  # array elements are values
                result.append(ch)
                i += 1
                continue

            if ch in ('}', ']'):
                if stack:
                    stack.pop()
                expect_value = False
                result.append(ch)
                i += 1
                continue

            if ch == ':':
                expect_value = True
                result.append(ch)
                i += 1
                continue

            if ch == ',':
                if stack and stack[-1] == 'object':
                    expect_value = False  # next token is a key
                else:
                    expect_value = True   # next token is a value (in array)
                result.append(ch)
                i += 1
                continue

            # Whitespace and other characters
            result.append(ch)
            i += 1

        return ''.join(result)

    # ── 清洗步骤 6：不完整 JSON 修复 ────────────────────────────────────────

    def _fix_incomplete_json(self, text):
        """
        修复截断/不完整的 JSON：
          1. 补全未关闭的字符串
          2. 处理末尾裸冒号（缺失值 → 补 null）
          3. 移除末尾尾随逗号
          4. 按逆序补全所有未关闭的括号
        """
        text = text.rstrip()
        if not text:
            return text

        stack = []
        in_string = False
        quote_char = None
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]
            if in_string:
                if ch == '\\' and i + 1 < n:
                    i += 2
                    continue
                if ch == quote_char:
                    in_string = False
            else:
                if ch in ('"', "'"):
                    in_string = True
                    quote_char = ch
                elif ch in self._OPENING:
                    stack.append(ch)
                elif ch in self._CLOSING_CHARS:
                    if stack:
                        stack.pop()
            i += 1

        # 修复1：补全未关闭的字符串
        if in_string:
            text += '"'

        # 修复2：处理末尾特殊情况
        stripped = text.rstrip()
        if stripped.endswith(':'):
            text = stripped + ' null'
        elif stripped.endswith(','):
            text = stripped[:-1].rstrip()

        # 修复3：补全缺失的括号
        for bracket in reversed(stack):
            text += self._CLOSING[bracket]

        return text


# ── 模块级便捷函数 ────────────────────────────────────────────────────────────

_default_parser = LLMJsonParser()


def parse(text):
    """模块级 parse 快捷函数，等同于 LLMJsonParser().parse(text)。"""
    return _default_parser.parse(text)


def parse_safe(text, default=None):
    """模块级 parse_safe 快捷函数，失败返回 default。"""
    return _default_parser.parse_safe(text, default=default)


def parse_all(text):
    """模块级 parse_all 快捷函数，返回所有 JSON 对象列表。"""
    return _default_parser.parse_all(text)
