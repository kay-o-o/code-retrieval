"""Semantic code search for mini-coding-agent.

Why this exists
---------------
Upstream `search` is rg / `pattern in line`. That only works when the question
already uses the same words as the source. A Chinese question about a numeric
limit cannot match `MAX_TOOL_OUTPUT = 4000`, so the agent burns its whole step
budget and gives up. Measured on 10 in-repo tasks: 3 of 10 failures were
exactly this, all of them "Chinese question, English identifier".

How it works
------------
1. Split every text file into overlapping chunks (line ranges, so results stay
   citable as `path:start-end`).
2. Build two rankings per query:
   - BM25 over identifiers + CJK bigrams (sharp on symbol names)
   - cosine over embeddings (sharp on paraphrases / cross-lingual)
3. Fuse with weighted RRF (vector 0.7 / keyword 0.3). Equal weights were tried
   first: recall was fine but top-1 dropped, same finding as the cs-agent run.

Vector cache lives in `.mini-coding-agent/code_index.json` and is keyed by a
hash of the chunk text, so re-indexing an unchanged repo is free and an edited
file only re-embeds its own chunks.

Zero third-party dependencies on purpose: urllib for HTTP, math for cosine.
"""

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

INDEX_VERSION = 1
DEFAULT_EMBED_MODEL = "text-embedding-v3"
EMBED_BATCH = 10

TEXT_SUFFIXES = {
    ".py", ".pyi", ".md", ".markdown", ".txt", ".toml", ".cfg", ".ini",
    ".json", ".yml", ".yaml", ".sh", ".bash", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".go", ".rs", ".c", ".h", ".cpp", ".sql", ".html", ".css",
}
SKIP_DIRS = {
    ".git", ".mini-coding-agent", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".venv", "venv", "node_modules", "dist", "build", ".mypy_cache", ".idea",
}
MAX_FILE_BYTES = 200_000
CONFIG_JSON_NAMES = {
    "package.json", "tsconfig.json", "composer.json", "manifest.json",
    "openapi.json", "swagger.json", "langgraph.json", "app.json", "now.json",
    "vercel.json", "babel.json", ".eslintrc", "launch.json",
}

K1, B = 1.5, 0.75
# RRF k is not a universal constant. The usual 60 comes from web-scale
# collections; with ~150 chunks it flattens every rank into nearly the same
# weight and a correct hit gets diluted. Re-tuned on the 11-case suite once the
# gold went from line numbers to anchors (the old numbers in this comment were
# measured against stale offsets and were simply wrong): k=10 with an 0.8/0.2
# split gave R@3 36.4% / MRR 0.276, versus 27.3% / 0.200 at k=3 with 0.5/0.5.
# Keyword weight stays non-zero because it is the only path that survives a
# typo'd identifier, not because it lifts recall on paraphrased Chinese.
RRF_K = 10
W_VECTOR, W_KEYWORD = 0.8, 0.2
# A term appearing in at most this many chunks is treated as evidence, not noise.
RARE_DF = 3
EXACT_BONUS = 0.008
# Ranking a definition above a mention. Tuned separately from the weights
# because it fixes a bias the two rankers share rather than trading them off.
DEFINITION_BONUS = 0.02

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


class SemanticSearchError(RuntimeError):
    """Raised when the index cannot be built (no key, network, bad response)."""


def tokenize(text):
    """Identifiers + snake_case parts + CJK bigrams.

    CJK has no spaces, so bigrams are the cheapest stand-in for a segmenter and
    they avoid pulling in jieba.
    """
    lowered = text.lower()
    terms = []
    for word in _WORD_RE.findall(lowered):
        if len(word) > 1:
            terms.append(word)
        for part in word.split("_"):
            if len(part) > 2 and part != word:
                terms.append(part)
    for run in _CJK_RE.findall(lowered):
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[i:i + 2] for i in range(len(run) - 1))
    return terms


def has_cjk(text):
    return bool(_CJK_CHAR_RE.search(text))


class Chunk:
    __slots__ = ("path", "start", "end", "text", "context")

    def __init__(self, path, start, text, context=""):
        self.path = path
        self.start = start
        self.text = text
        self.context = context
        self.end = start + text.count("\n")

    @property
    def key(self):
        return f"{self.path}:{self.start}-{self.end}"

    @property
    def embed_text(self):
        """What gets embedded: the chunk plus where it lives.

        Contextual retrieval. A constants block reads as `MAX_TOOL_OUTPUT =
        4000` and nothing else, so on its own it is a poor match for a question
        phrased in another language. Prefixing the file and the enclosing
        symbol gives the embedder something to attach to, at zero query cost.
        Measured on T08: the target chunk moved from rank 33 to top-3.
        """
        if not self.context:
            return self.text
        return f"{self.context}\n{self.text}"

    @property
    def digest(self):
        payload = self.embed_text.encode("utf-8", "replace")
        return hashlib.sha1(payload).hexdigest()


_SYMBOL_RE = re.compile(r"^\s*(?:async\s+)?(def|class)\s+([A-Za-z_]\w*)")


def _symbol_map(lines):
    """For every line, the innermost def/class that is in scope at that line."""
    symbols, current = [], ""
    for line in lines:
        match = _SYMBOL_RE.match(line)
        if match:
            current = f"{match.group(1)} {match.group(2)}"
        symbols.append(current)
    return symbols


def _span_label(symbols, limit=4):
    """Name every scope a chunk covers, not just the one it starts in.

    A chunk opens with up to six overlap lines from the previous function, so
    `symbols[0]` often names code that is not the chunk's subject. The read_file
    schema chunk began with the tail of `remember` and was therefore labelled
    `def remember`; a question about read_file could not find it (measured: T06
    missed the top-5 entirely). Listing the whole span fixes the label without
    touching the cut points, which were already tuned for recall.
    """
    seen = []
    for symbol in symbols:
        if symbol and symbol not in seen:
            seen.append(symbol)
    if not seen:
        return "模块级"
    if len(seen) > limit:
        return " / ".join(seen[:limit]) + f" 等 {len(seen)} 个符号"
    return " / ".join(seen)


def _split_markdown(rel, text, max_lines=40):
    chunks, current, start = [], [], 1
    heading = ""
    for number, line in enumerate(text.splitlines(), start=1):
        is_heading = line.startswith("#")
        if is_heading:
            heading = line.lstrip("#").strip()
        if is_heading and len(current) >= 6:
            body = "\n".join(current)
            if body.strip():
                chunks.append(Chunk(rel, start, body, f"文档 {rel} › {heading}"))
            current, start = [], number
        current.append(line)
        if len(current) >= max_lines:
            body = "\n".join(current)
            if body.strip():
                chunks.append(Chunk(rel, start, body, f"文档 {rel} › {heading}"))
            current, start = [], number + 1
    if current and "\n".join(current).strip():
        chunks.append(Chunk(rel, start, "\n".join(current), f"文档 {rel} › {heading}"))
    return chunks


def _split_code(rel, text, target=26, overlap=6):
    """Cut at a blank line once a chunk is long enough, keeping `overlap` lines.

    Blank-line cuts usually land between functions. The break window starts at
    two thirds of the target instead of halfway: cutting at the first blank line
    after the midpoint produced ~13-line chunks that tore functions apart and
    left the retrieval target outside the top-3.
    """
    lines = text.splitlines()
    if not lines:
        return []
    symbols = _symbol_map(lines)
    chunks, cursor = [], 0
    while cursor < len(lines):
        stop = min(cursor + target, len(lines))
        if stop < len(lines):
            for probe in range(cursor + target * 2 // 3, stop):
                if not lines[probe].strip():
                    stop = probe + 1
                    break
        body = "\n".join(lines[cursor:stop])
        if body.strip():
            chunks.append(Chunk(rel, cursor + 1, body, f"文件 {rel} › {_span_label(symbols[cursor:stop])}"))
        if stop >= len(lines):
            break
        cursor = max(stop - overlap, cursor + 1)
    return chunks


def split_file(rel, text):
    if len(text) > MAX_FILE_BYTES:
        return []
    if rel.endswith((".md", ".markdown")):
        return _split_markdown(rel, text)
    return _split_code(rel, text)


def _is_data_dump(path):
    """Config JSON is worth indexing; data dumps are poison.

    Measured problem: `eval/*.json` result files contain the evaluation
    questions verbatim, so a semantic query matched the dump instead of the
    source it asked about, and the agent read the wrong file. Only JSON with a
    config-like name is indexed; anything else with a `.json` suffix is treated
    as exported data.
    """
    if path.suffix.lower() != ".json":
        return False
    name = path.name.lower()
    if name in CONFIG_JSON_NAMES:
        return False
    if name.endswith(("config.json", "settings.json", ".rc.json", "schema.json")):
        return False
    return True


def collect_chunks(root):
    root = Path(root)
    chunks = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _is_data_dump(path):
            continue
        chunks.extend(split_file(relative.as_posix(), text))
    return chunks


def _normalize(vector):
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class EmbeddingClient:
    """Minimal OpenAI-compatible /embeddings caller."""

    def __init__(self, api_key=None, base_url=None, model=None, timeout=30):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("EMBED_MODEL") or DEFAULT_EMBED_MODEL
        self.timeout = timeout

    def available(self):
        return bool(self.api_key)

    def embed(self, texts):
        if not self.api_key:
            raise SemanticSearchError("no OPENAI_API_KEY for embeddings")
        vectors = []
        for offset in range(0, len(texts), EMBED_BATCH):
            vectors.extend(self._embed_batch(texts[offset:offset + EMBED_BATCH]))
        return vectors

    def _embed_batch(self, batch):
        payload = {"model": self.model, "input": batch, "encoding_format": "float"}
        request = urllib.request.Request(
            self.base_url + "/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:200]
            raise SemanticSearchError(f"embeddings HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise SemanticSearchError(f"embeddings unavailable: {error}") from error

        items = data.get("data") or []
        if len(items) != len(batch):
            raise SemanticSearchError("embeddings returned wrong item count")
        items.sort(key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items]


class CodeIndex:
    def __init__(self, root, client=None, cache_path=None, contextual=True):
        self.root = Path(root)
        self.contextual = contextual
        self.client = client or EmbeddingClient()
        self.cache_path = Path(cache_path) if cache_path else (
            self.root / ".mini-coding-agent" / "code_index.json"
        )
        self.chunks = []
        self.vectors = []
        self._built = False
        self.build_stats = {}

    # ---------------------------- build ----------------------------
    def _load_cache(self):
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if (
            raw.get("version") != INDEX_VERSION
            or raw.get("model") != self.client.model
            or raw.get("contextual", True) != self.contextual
        ):
            return {}
        return raw.get("vectors") or {}

    def _save_cache(self, cache):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "model": self.client.model,
            "contextual": self.contextual,
            "vectors": cache,
        }
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.cache_path)

    def build(self, force=False):
        chunks = collect_chunks(self.root)
        cache = {} if force else self._load_cache()

        missing = [chunk for chunk in chunks if chunk.digest not in cache]
        embedded = 0
        if missing:
            texts = [chunk.embed_text if self.contextual else chunk.text for chunk in missing]
            fresh = self.client.embed(texts)
            for chunk, vector in zip(missing, fresh):
                cache[chunk.digest] = [round(value, 5) for value in _normalize(vector)]
            embedded = len(missing)

        self.chunks = chunks
        self.vectors = [cache[chunk.digest] for chunk in chunks]
        self._terms = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self._doc_freq = Counter()
        for terms in self._terms:
            self._doc_freq.update(terms.keys())
        self._avg_len = sum(sum(c.values()) for c in self._terms) / max(len(chunks), 1)
        self._built = True
        self.build_stats = {
            "chunks": len(chunks),
            "embedded": embedded,
            "cached": len(chunks) - embedded,
        }
        if embedded:
            self._save_cache(cache)
        return self

    def ensure(self):
        if not self._built:
            self.build()
        return self

    # ---------------------------- search ----------------------------
    def _bm25_scores(self, query_terms):
        total = len(self.chunks)
        average = self._avg_len or 1
        scores = []
        for counts in self._terms:
            length = sum(counts.values()) or 1
            score = 0.0
            for term in query_terms:
                freq = counts.get(term)
                if not freq:
                    continue
                idf = math.log(1 + (total - self._doc_freq[term] + 0.5) / (self._doc_freq[term] + 0.5))
                score += idf * freq * (K1 + 1) / (freq + K1 * (1 - B + B * length / average))
            scores.append(score)
        return scores

    def _rare_terms(self, query_terms):
        """Query terms selective enough to be evidence, not noise.

        Rank fusion is blind to decisiveness: a query naming an identifier has
        one obviously right chunk, but if a semantically vague chunk scores a
        better vector rank it still wins. Same failure the cs-agent run hit.
        Counting how many rare query terms a chunk actually contains restores
        the signal a human grepping would use.
        """
        seen, rare = set(), []
        for term in query_terms:
            if term in seen:
                continue
            seen.add(term)
            if len(term) >= 3 and self._doc_freq.get(term, 0) <= RARE_DF:
                rare.append(term)
        return rare

    def _definition_hits(self, terms):
        """Chunks that define one of `terms`, not merely mention it.

        Both rankers systematically prefer the mention over the definition: ask
        for `tool_write_file` and BM25 returns the tool-schema entry that lists
        the name, while the vector ranker returns the schema too. The chunk the
        question is actually about is `def tool_write_file` further down. Same
        for `MAX_TOOL_OUTPUT` (schema mention beats the assignment). A human
        grepping would go to the definition, so the fusion gets the same prior.

        A definition is a line whose stripped text opens with the identifier and
        continues into `=`, `(`, `:` or `.`; `def X` / `class X` count too, and so
        does a function whose name merely contains X (`def tool_read_file` for
        the query `read_file`), plus a quoted schema key (`"read_file": {`).
        The looser forms are what make the tool-schema blocks reachable at all.
        """
        patterns = []
        for term in terms:
            if not re.fullmatch(r"[A-Za-z_]\w{2,}", term):
                continue
            escaped = re.escape(term)
            patterns.append(
                re.compile(
                    rf"""^\s*(?:
                        (?:def|class)\s+\w*{escaped}\b
                      | ["']?{escaped}["']?\s*[=:(.]
                    )""",
                    re.VERBOSE,
                )
            )
        if not patterns:
            return {}
        hits = {}
        for index, chunk in enumerate(self.chunks):
            for line in chunk.text.splitlines():
                if any(pattern.match(line) for pattern in patterns):
                    hits[index] = hits.get(index, 0) + 1
        return hits

    def search(self, query, top_k=4, path_filter=None):
        self.ensure()
        if not self.chunks:
            return []

        query_terms = tokenize(query)
        keyword_scores = self._bm25_scores(query_terms)
        keyword_order = sorted(range(len(self.chunks)), key=lambda i: -keyword_scores[i])

        query_vector = _normalize(self.client.embed([query])[0])
        vector_scores = [
            sum(a * b for a, b in zip(query_vector, vector)) for vector in self.vectors
        ]
        vector_order = sorted(range(len(self.chunks)), key=lambda i: -vector_scores[i])

        rare = self._rare_terms(query_terms)

        fused = {}
        for rank, index in enumerate(keyword_order):
            fused[index] = fused.get(index, 0.0) + W_KEYWORD / (RRF_K + rank + 1)
        for rank, index in enumerate(vector_order):
            fused[index] = fused.get(index, 0.0) + W_VECTOR / (RRF_K + rank + 1)
        if rare:
            for index, counts in enumerate(self._terms):
                hits = sum(1 for term in rare if counts.get(term))
                if hits:
                    fused[index] = fused.get(index, 0.0) + EXACT_BONUS * hits

        for index, hits in self._definition_hits(query_terms).items():
            fused[index] = fused.get(index, 0.0) + DEFINITION_BONUS * hits

        ranked = sorted(fused.items(), key=lambda item: -item[1])
        results = []
        for index, score in ranked:
            chunk = self.chunks[index]
            if path_filter and not chunk.path.startswith(path_filter):
                continue
            results.append((chunk, score))
            if len(results) >= top_k:
                break
        return results


def format_hits(results, query, max_lines=16, max_chars=2600):
    """Render hits as `path:start-end` plus a line window from each.

    The character budget is shared out across all hits instead of being spent
    first-come: the old loop `break`ed once the budget was gone, which silently
    dropped every hit after the second one. A wrong top-1 then made the answer
    unreachable no matter how well the rest of the ranking did (T10: the retry
    branch sat at rank 4 and was never shown). Shrinking each window to keep
    every hit visible is strictly better than showing two hits in full.
    """
    if not results:
        return "(no matches)"

    query_terms = tokenize(query)
    blocks, used = [], 0
    for position, (chunk, score) in enumerate(results):
        lines = chunk.text.splitlines()
        best, best_hits = 0, 0
        for offset, line in enumerate(lines):
            lowered = line.lower()
            hits = sum(1 for term in query_terms if term in lowered)
            if hits > best_hits:
                best, best_hits = offset, hits
        remaining = len(results) - position
        allowance = max(4, min(max_lines, (max_chars - used) // max(remaining, 1) // 60))
        start = max(0, min(best - 2, len(lines) - allowance))
        window = lines[start:start + allowance]
        body = "\n".join(
            f"{chunk.start + start + offset:>5}: {line}"
            for offset, line in enumerate(window)
        )
        block = f"### {chunk.key}  (score {score:.4f})\n{body}"
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def looks_like_literal(pattern):
    """True when rg is likely to succeed: an identifier, not a sentence."""
    stripped = pattern.strip()
    if not stripped or has_cjk(stripped):
        return False
    if " " in stripped or len(stripped) > 40:
        return False
    return bool(re.fullmatch(r"[\w.\-*/\\|^$()\[\]{}?+*]+", stripped))
