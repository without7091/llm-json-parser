# LLM JSON Parser

从大模型（LLM）回答中健壮地提取和修复 JSON。

## 解决的问题

大模型返回的 JSON 往往带有各种"噪声"，导致 `json.loads()` 直接失败。本工具通过多阶段流水线 + 渐进式清洗解决这些问题：

| 问题 | 示例 |
|------|------|
| Markdown 代码块包裹 | `` ```json {...} ``` `` |
| JSON 前后有额外文本 | `"结果如下：{...}，请查看"` |
| Python 风格字面量 | `True / False / None` |
| JS 字面量 | `undefined` |
| JavaScript 注释 | `// comment` 和 `/* comment */` |
| 尾随逗号 | `{"a": 1, "b": 2,}` |
| 单引号代替双引号 | `{'key': 'value'}` |
| 截断/不完整的 JSON | `{"name": "Alice", "items": [1, 2` |
| 多个 JSON 块 | 一段文本中包含多个 JSON 对象 |

## 特性

- **纯标准库**：仅依赖 `json`、`re`、`sys`，无第三方依赖
- **广泛兼容**：Python 2.7+ 和 Python 3.x 全系列
- **渐进式清洗**：每步清洗后立即尝试解析，用最少干预得到有效结果
- **单文件**：直接复制 `llm_json_parser.py` 即可使用

## 安装

直接复制 `llm_json_parser.py` 到你的项目中：

```bash
cp llm_json_parser.py /path/to/your/project/
```

## 快速使用

```python
from llm_json_parser import LLMJsonParser

parser = LLMJsonParser()

# 从 LLM 输出中提取 JSON
result = parser.parse(llm_output)

# 安全解析（失败返回默认值）
result = parser.parse_safe(llm_output, default={})

# 提取文本中所有 JSON
results = parser.parse_all(llm_output)
```

也可以使用模块级快捷函数：

```python
import llm_json_parser

result = llm_json_parser.parse(llm_output)
result = llm_json_parser.parse_safe(llm_output, default={})
results = llm_json_parser.parse_all(llm_output)
```

## 使用示例

### Markdown 代码块 + Python 字面量

```python
text = '''
```json
{"users": [{"name": "Alice", "active": True}], "total": 1,}
```
'''
result = parser.parse(text)
# → {"users": [{"name": "Alice", "active": true}], "total": 1}
```

### 单引号 + 注释 + 尾随逗号

```python
text = "{'tags': ['a', 'b',], /* note */ 'count': None}"
result = parser.parse(text)
# → {"tags": ["a", "b"], "count": null}
```

### 截断修复

```python
text = '{"name": "Alice", "items": [1, 2, 3'
result = parser.parse(text)
# → {"name": "Alice", "items": [1, 2, 3]}
```

### 从文本中提取

```python
text = '分析结果如下：{"score": 85, "level": "优秀"}，建议继续保持。'
result = parser.parse(text)
# → {"score": 85, "level": "优秀"}
```

## 处理流程

```
输入文本
  ├─ 阶段0：直接 json.loads()（快速路径）
  ├─ 阶段1：从 Markdown 代码块提取
  ├─ 阶段2：括号匹配算法定位 JSON 边界
  └─ 阶段3：渐进式清洗（每步后立即尝试解析）
              Step 1: 去除 JS 注释
              Step 2: 替换 Python/JS 字面量
              Step 3: 去除尾随逗号
              Step 4: 单引号 → 双引号
              Step 5: 不完整 JSON 修复（最后手段）
```

## 测试

```bash
python -m pytest test_core.py test_llm_scenarios.py -v
```

- `test_core.py`：核心功能与各清洗步骤的单元测试
- `test_llm_scenarios.py`：模拟 GPT / Claude / Gemini / 通义千问 / 文心一言等大模型的真实输出场景

## 已知局限

| 场景 | 原因 |
|------|------|
| `{'text': 'it's tricky'}` 裸撇号 | 语法歧义，无法可靠区分 |
| `{key: value}` 无引号键 | 需完整解析器，有误伤风险 |
| `NaN` / `Infinity` | 行为取决于 Python json 模块实现 |
| 键名中间截断 `{"ke` | 无法推断完整键名 |

## License

MIT
