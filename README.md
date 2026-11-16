# code-retrieval

**给一个开源 coding agent 换掉检索层：子串匹配 → BM25 + 向量混合检索。端到端任务成功率 2/10 → 10/10。**

[![CI](https://github.com/kay-o-o/code-retrieval/actions/workflows/ci.yml/badge.svg)](https://github.com/kay-o-o/code-retrieval/actions/workflows/ci.yml)

> fork 自 [rasbt/mini-coding-agent](https://github.com/rasbt/mini-coding-agent)（Apache-2.0），**不是原创项目**。
> 改动清单见 [NOTICE](NOTICE)，逐轮排查过程见 [docs/改造记录.md](docs/改造记录.md)。

## 问题

上游 `search` 只有一行子串匹配。中文提问碰上英文标识符，必然 0 命中：

| 提问 | 应为 | 上游 `search` 返回 |
|---|---|---|
| 工具返回内容超过多少字符截断 | `MAX_TOOL_OUTPUT = 4000` | `(no matches)` |
| `read_file` 一次最多读多少行 | `int=200` | `(no matches)` |

模型于是换词重搜，把步数烧完。10 道仓库内任务只过 2 道，失败形态统一是 `Stopped after reaching the step limit`。

## 改了三处

| 层 | 上游做法 | 现在 |
|---|---|---|
| 工具调用 | prompt 教模型写 `<tool>` 标签，正则从正文解析 | 原生 function calling，走 `tool_calls` 通道 |
| 轮次收口 | 模型只能输出 `<tool>` 或 `<final>`，天然会停 | 预算耗尽轮不再下发 `tools`，由代码侧强制作答 |
| 检索 | 子串匹配 | BM25 + 向量加权 RRF（k=10，0.8/0.2），标识符定义处加权，块内补全符号上下文头 |

三处缺一不可：只接原生协议仍是 2/10，只加强制收口是 7/10，检索改完才到 10/10。

## 效果

端到端 10 题（deepseek-v4.1-flash，`max_steps=6`）：

| 阶段 | 通过 | 平均步数 |
|---|---|---|
| 上游原样 | 2/10 | 5.3 |
| + 原生 function calling | 2/10 | 5.4 |
| + 强制收口 | 7/10 | 5.0 |
| + 语义检索 | 8/10 | 5.1 |
| + 上下文头修正 | 9/10 | 4.5 |
| + 定义优先 & 作答预算 | **10/10** | 5.0 |

单题平均 9.5 秒，全套约 95 秒。

检索单测（11 条用例，top_k=5）：

| 方式 | R@1 | R@3 | R@5 | MRR |
|---|---|---|---|---|
| 子串匹配 | 0.0% | 0.0% | 0.0% | 0.000 |
| BM25 单路 | 18.2% | 18.2% | 27.3% | 0.205 |
| 向量单路 | 9.1% | 36.4% | 45.5% | 0.250 |
| **混合** | **18.2%** | **36.4%** | **54.5%** | **0.294** |

## 跑起来

```bash
git clone https://github.com/kay-o-o/code-retrieval.git
cd code-retrieval
```

运行时零第三方依赖，Python 3.10+：

```bash
# 本地 Ollama（默认后端）
ollama serve & ollama pull qwen2.5:3b
python mini_coding_agent.py

# 任意 OpenAI 兼容端点
cp .env.example .env        # 填 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
python mini_coding_agent.py --provider openai
```

评测与单测：

```bash
python -m pytest -q                                  # 上游单测，18 passed / 1 skipped
python scripts/eval_baseline.py --provider openai   # 端到端 10 题，约 95 秒
python scripts/eval_retrieval.py                    # 检索 11 条，约 20 秒
```

> `--approval` 默认 `ask`，写文件、跑 shell 前会先问。`auto` 只在可信仓库里用。

## 目录

```
mini_coding_agent.py      主程序，1,378 行（上游 1,019 行）
code_index.py             索引与混合检索，568 行，新增
scripts/                  两套评测：端到端 / 检索
docs/改造记录.md           完整排查过程与全部中间数据
NOTICE                    上游版权声明与逐文件修改说明
```

## 许可

Apache License 2.0，见 [LICENSE](LICENSE)。上游版权归 [Sebastian Raschka](https://github.com/rasbt)，本仓库修改部分见 [NOTICE](NOTICE)。
