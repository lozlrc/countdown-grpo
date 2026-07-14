"""Countdown task: puzzle generation, prompt rendering, and reward scoring.

A puzzle gives N numbers (3 or 4) and a target. A correct answer is an
arithmetic expression over +, -, *, / and parentheses that uses each given
number exactly once and evaluates to the target.
"""

from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass
from fractions import Fraction

MAX_EXPR_LEN = 256

# Intermediate values during puzzle construction are kept in this range so
# every generated puzzle has at least one all-integer derivation.
_INTERMEDIATE_MAX = 9999


@dataclass(frozen=True)
class Puzzle:
    nums: tuple[int, ...]
    target: int
    solution: str  # one known valid expression, kept for debugging/eval sanity

    @property
    def key(self) -> tuple[tuple[int, ...], int]:
        return (tuple(sorted(self.nums)), self.target)


# ---------------------------------------------------------------------------
# Puzzle generation
# ---------------------------------------------------------------------------


def _compose(nums: list[int], rng: random.Random) -> tuple[int, str] | None:
    """Combine all numbers with random binary ops; return (value, expression).

    Only integer-valued intermediates are allowed (subtraction must stay
    positive, division must be exact), which guarantees reachability of the
    final target by construction.
    """
    items: list[tuple[int, str]] = [(n, str(n)) for n in nums]
    while len(items) > 1:
        i, j = rng.sample(range(len(items)), 2)
        (a, ea), (b, eb) = items[i], items[j]
        candidates: list[tuple[int, str]] = [
            (a + b, f"({ea} + {eb})"),
            (a * b, f"({ea} * {eb})"),
        ]
        if a > b:
            candidates.append((a - b, f"({ea} - {eb})"))
        elif b > a:
            candidates.append((b - a, f"({eb} - {ea})"))
        if b != 0 and a % b == 0:
            candidates.append((a // b, f"({ea} / {eb})"))
        elif a != 0 and b % a == 0:
            candidates.append((b // a, f"({eb} / {ea})"))
        candidates = [(v, e) for v, e in candidates if 1 <= v <= _INTERMEDIATE_MAX]
        if not candidates:
            return None
        val, expr = rng.choice(candidates)
        items = [it for k, it in enumerate(items) if k not in (i, j)]
        items.append((val, expr))
    value, expr = items[0]
    # Strip the single outermost parenthesis pair for readability.
    if expr.startswith("(") and expr.endswith(")"):
        expr = expr[1:-1]
    return value, expr


def generate_puzzle(
    rng: random.Random,
    num_count: int,
    nums_min: int = 1,
    nums_max: int = 99,
    target_min: int = 10,
    target_max: int = 99,
) -> Puzzle:
    """Rejection-sample a puzzle whose target is reachable by construction."""
    while True:
        nums = [rng.randint(nums_min, nums_max) for _ in range(num_count)]
        result = _compose(nums, rng)
        if result is None:
            continue
        target, solution = result
        if not (target_min <= target <= target_max):
            continue
        if target in nums:  # avoid answers that trivially echo one input
            continue
        return Puzzle(nums=tuple(nums), target=target, solution=solution)


def make_dataset(
    n: int,
    seed: int,
    num_counts: tuple[int, ...] = (3, 4),
    exclude: set[tuple[tuple[int, ...], int]] | None = None,
    **kwargs,
) -> list[Puzzle]:
    """Generate n unique puzzles, deduped by (sorted nums, target)."""
    rng = random.Random(seed)
    seen: set[tuple[tuple[int, ...], int]] = set(exclude) if exclude else set()
    puzzles: list[Puzzle] = []
    while len(puzzles) < n:
        p = generate_puzzle(rng, rng.choice(num_counts), **kwargs)
        if p.key in seen:
            continue
        seen.add(p.key)
        puzzles.append(p)
    return puzzles


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

TEMPLATES = ("plain", "chat", "r1")

_TASK = (
    "Using the numbers {nums}, create an equation that equals {target}. "
    "You may use +, -, *, / and parentheses, and each number must be used "
    "exactly once. Show your reasoning in <think> </think> tags, then give "
    "only the final expression in <answer> </answer> tags, for example "
    "<answer> (1 + 2) / 3 </answer>."
)


def render_prompt(puzzle: Puzzle, template: str, tokenizer=None) -> tuple[str, bool]:
    """Render a prompt. Returns (text, think_open).

    think_open is True when the prompt itself ends with an opening <think>
    tag, i.e. the completion is generated from inside the think block.
    """
    task = _TASK.format(nums=list(puzzle.nums), target=puzzle.target)
    if template == "plain":
        return task + "\n", False
    if template == "chat":
        if tokenizer is None or tokenizer.chat_template is None:
            raise ValueError("chat template requires a tokenizer with a chat_template")
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": task}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return text, False
    if template == "r1":
        text = (
            "A conversation between User and Assistant. The User asks a "
            "question, and the Assistant solves it. The Assistant first "
            "reasons about the problem, then gives the final answer.\n"
            f"User: {task}\n"
            "Assistant: Let me solve this step by step.\n<think>"
        )
        return text, True
    raise ValueError(f"unknown template {template!r}, expected one of {TEMPLATES}")


# ---------------------------------------------------------------------------
# Safe expression evaluation
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)


class _UnsafeExpression(ValueError):
    pass


def safe_eval(expr: str) -> tuple[Fraction, list[int]] | None:
    """Evaluate an arithmetic expression without executing anything.

    Walks the AST with a strict whitelist: integer literals and the four
    binary ops only. Names, calls, attributes, subscripts, unary ops, floats,
    and everything else are rejected. Division is exact via Fraction.

    Returns (value, leaf integers in order of appearance) or None if invalid.
    """
    if not expr or len(expr) > MAX_EXPR_LEN:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None

    used: list[int] = []

    def ev(node: ast.AST) -> Fraction:
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise _UnsafeExpression("division by zero")
            return left / right
        # bool is a subclass of int, so require exactly int
        if isinstance(node, ast.Constant) and type(node.value) is int:
            used.append(node.value)
            return Fraction(node.value)
        raise _UnsafeExpression(f"disallowed node {type(node).__name__}")

    try:
        value = ev(tree.body)
    except _UnsafeExpression:
        return None
    return value, used


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class Score:
    format_ok: bool
    answer_ok: bool
    reward: float


def extract_answer(text: str) -> str | None:
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else None


def check_format(text: str) -> bool:
    """Exactly one well-formed <think> block followed by one <answer> block."""
    if len(_THINK_RE.findall(text)) != 1 or len(_ANSWER_RE.findall(text)) != 1:
        return False
    think_end = text.find("</think>")
    answer_start = text.find("<answer>")
    return think_end != -1 and answer_start != -1 and think_end < answer_start


def check_answer(text: str, puzzle: Puzzle) -> bool:
    expr = extract_answer(text)
    if expr is None:
        return False
    result = safe_eval(expr)
    if result is None:
        return False
    value, used = result
    # each provided number used exactly once, as a multiset
    if sorted(used) != sorted(puzzle.nums):
        return False
    return value == puzzle.target


def score_completion(
    completion: str,
    puzzle: Puzzle,
    think_open: bool = False,
    mode: str = "shaped",
    format_reward: float = 0.1,
    answer_reward: float = 1.0,
) -> Score:
    """Tiered reward.

    mode="shaped": answer_reward if the answer is correct, else format_reward
    if the tag structure is right, else 0. mode="answer_only": answer_reward
    or 0, no format shaping. The shaped-vs-answer-only split is an ablation.
    """
    text = "<think>" + completion if think_open else completion
    format_ok = check_format(text)
    answer_ok = check_answer(text, puzzle)
    if mode == "answer_only":
        reward = answer_reward if answer_ok else 0.0
    elif mode == "shaped":
        if answer_ok:
            reward = answer_reward
        elif format_ok:
            reward = format_reward
        else:
            reward = 0.0
    else:
        raise ValueError(f"unknown reward mode {mode!r}")
    return Score(format_ok=format_ok, answer_ok=answer_ok, reward=reward)
