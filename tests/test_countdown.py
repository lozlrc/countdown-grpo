import random

import pytest

from grpo.countdown import (
    Puzzle,
    check_format,
    extract_answer,
    generate_puzzle,
    make_dataset,
    render_prompt,
    safe_eval,
    score_completion,
)


def wrap(expr: str) -> str:
    return f"<think>let me try</think> <answer>{expr}</answer>"


class TestGenerator:
    def test_dataset_properties(self):
        puzzles = make_dataset(50, seed=0)
        assert len({p.key for p in puzzles}) == 50
        for p in puzzles:
            assert len(p.nums) in (3, 4)
            assert all(1 <= n <= 99 for n in p.nums)
            assert 10 <= p.target <= 99
            assert p.target not in p.nums

    def test_generated_solution_is_valid(self):
        # the generator's own solution must pass the reward's validator
        puzzles = make_dataset(50, seed=1)
        for p in puzzles:
            result = safe_eval(p.solution)
            assert result is not None, p
            value, used = result
            assert value == p.target
            assert sorted(used) == sorted(p.nums)

    def test_exclude_keys(self):
        first = make_dataset(20, seed=2)
        second = make_dataset(20, seed=2, exclude={p.key for p in first})
        assert not {p.key for p in first} & {p.key for p in second}

    def test_custom_ranges(self):
        rng = random.Random(0)
        p = generate_puzzle(rng, 3, nums_min=1, nums_max=9, target_min=5, target_max=20)
        assert all(1 <= n <= 9 for n in p.nums)
        assert 5 <= p.target <= 20


class TestSafeEval:
    def test_basic_arithmetic(self):
        value, used = safe_eval("(2 + 3) * 4")
        assert value == 20
        assert sorted(used) == [2, 3, 4]

    def test_exact_division(self):
        # 7/2 is not an integer; Fraction arithmetic must still land exactly
        value, used = safe_eval("7 / 2 * 4")
        assert value == 14

    def test_inexact_result_not_rounded(self):
        value, _ = safe_eval("10 / 3")
        assert value != 3
        assert value != 3.3333333333333335

    def test_division_by_zero(self):
        assert safe_eval("5 / (3 - 3)") is None

    @pytest.mark.parametrize(
        "expr",
        [
            "__import__('os').system('echo pwned')",
            "getattr(1, 'x')",
            "(5).__class__",
            "5 .__class__.__bases__",
            "[1, 2][0]",
            "{1: 2}[1]",
            "1 if True else 2",
            "2 ** 100",
            "5 % 3",
            "5 // 3",
            "-3 + 5",  # unary ops rejected: no way to smuggle extra numbers
            "1.5 * 2",
            "True + 3",
            "open('/etc/passwd')",
            "x + 1",
            "lambda: 1",
            "(1).bit_length()",
            "1e100",
            "",
        ],
    )
    def test_rejects_unsafe_or_disallowed(self, expr):
        assert safe_eval(expr) is None

    def test_rejects_overlong_expression(self):
        assert safe_eval("1" + " + 1" * 200) is None


class TestReward:
    puzzle = Puzzle(nums=(2, 3, 4), target=20, solution="(2 + 3) * 4")

    def test_correct_answer_full_reward(self):
        s = score_completion(wrap("(2 + 3) * 4"), self.puzzle)
        assert s.answer_ok and s.format_ok
        assert s.reward == 1.0

    def test_reuse_number_rejected(self):
        # 4 * 4 + 4 = 20 but uses 4 three times and skips 2 and 3
        s = score_completion(wrap("4 * 4 + 4"), self.puzzle)
        assert not s.answer_ok
        assert s.reward == 0.1  # format still fine

    def test_omitted_number_rejected(self):
        # (2 + 3) alone isn't the target; 5 * 4 = 20 omits nothing but
        # introduces 5, which is not in the puzzle's numbers
        assert not score_completion(wrap("5 * 4"), self.puzzle).answer_ok

    def test_wrong_value_rejected(self):
        assert not score_completion(wrap("2 + 3 + 4"), self.puzzle).answer_ok

    def test_injection_gets_no_reward(self):
        s = score_completion(wrap("__import__('os').getcwd()"), self.puzzle)
        assert not s.answer_ok

    def test_malformed_tags(self):
        # unclosed answer tag: no extractable answer, no format credit
        s = score_completion("<think>x</think><answer>(2+3)*4", self.puzzle)
        assert not s.format_ok and not s.answer_ok
        assert s.reward == 0.0

    def test_missing_think_tag(self):
        s = score_completion("<answer>(2 + 3) * 4</answer>", self.puzzle)
        assert not s.format_ok
        assert s.answer_ok  # answer validity is independent of format
        assert s.reward == 1.0

    def test_no_tags_zero_reward(self):
        s = score_completion("the answer is (2 + 3) * 4", self.puzzle)
        assert s.reward == 0.0

    def test_duplicate_answer_tags_break_format(self):
        text = "<think>x</think><answer>1</answer><answer>(2+3)*4</answer>"
        s = score_completion(text, self.puzzle)
        assert not s.format_ok
        assert s.answer_ok  # last answer tag is scored

    def test_think_open_prepends_tag(self):
        completion = "reasoning</think><answer>(2 + 3) * 4</answer>"
        assert not score_completion(completion, self.puzzle, think_open=False).format_ok
        s = score_completion(completion, self.puzzle, think_open=True)
        assert s.format_ok and s.reward == 1.0

    def test_answer_only_mode_ignores_format(self):
        s = score_completion(wrap("2 + 3"), self.puzzle, mode="answer_only")
        assert s.reward == 0.0
        s = score_completion(wrap("(2 + 3) * 4"), self.puzzle, mode="answer_only")
        assert s.reward == 1.0

    def test_extract_answer_takes_last(self):
        assert extract_answer("<answer>1</answer> <answer>2</answer>") == "2"
        assert extract_answer("no tags") is None

    def test_check_format_ordering(self):
        assert check_format("<think>a</think><answer>b</answer>")
        assert not check_format("<answer>b</answer><think>a</think>")


class TestPrompts:
    puzzle = Puzzle(nums=(2, 3, 4), target=20, solution="(2 + 3) * 4")

    def test_plain(self):
        text, think_open = render_prompt(self.puzzle, "plain")
        assert "[2, 3, 4]" in text and "20" in text
        assert not think_open

    def test_r1_ends_inside_think(self):
        text, think_open = render_prompt(self.puzzle, "r1")
        assert text.endswith("<think>")
        assert think_open

    def test_unknown_template(self):
        with pytest.raises(ValueError):
            render_prompt(self.puzzle, "nope")
