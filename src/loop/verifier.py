"""Answer Verification — robust parsing and comparison for model outputs.

Handles the messy reality of LLM answers: extracts the final answer from
chain-of-thought responses, normalizes formats, and compares against ground
truth with appropriate tolerance.
"""

import re
import math
from dataclasses import dataclass


@dataclass
class VerificationResult:
    is_correct: bool
    parsed_answer: str
    expected_answer: str
    raw_response: str


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(response: str) -> str | None:
    """Extract the final answer from a chain-of-thought response.

    Looks for patterns like:
        ANSWER: 42
        Answer: 42
        The answer is 42
        **Answer: 42**
        Final answer: 42
        = 42 (at end of line)

    Returns None if no answer pattern is found.
    """
    if not response or not response.strip():
        return None

    text = response.strip()

    # Pattern 1: "ANSWER: <value>" (our explicit prompt format)
    m = re.search(r'(?i)\bANSWER\s*[:=]\s*(.+)', text)
    if m:
        return _clean_answer(m.group(1))

    # Pattern 2: "Final answer: <value>" or "The final answer is <value>"
    m = re.search(r'(?i)(?:the\s+)?final\s+answer\s+(?:is|[:=])\s*(.+)', text)
    if m:
        return _clean_answer(m.group(1))

    # Pattern 3: "The answer is <value>"
    m = re.search(r'(?i)the\s+answer\s+is\s+(.+?)\.?\s*$', text, re.MULTILINE)
    if m:
        return _clean_answer(m.group(1))

    # Pattern 4: "= <value>" at end of a line — take the LAST match (final computation)
    matches = re.findall(r'=\s*(.+?)\s*$', text, re.MULTILINE)
    if matches:
        candidate = _clean_answer(matches[-1])
        if re.match(r'^-?[\d.,\s\[\]()]+$', candidate) or candidate.lower() in ('true', 'false'):
            return candidate

    # Pattern 5: Last line of the response (desperate fallback)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last = lines[-1]
        # Remove markdown bold
        last = re.sub(r'\*\*(.+?)\*\*', r'\1', last)
        # If last line is short and looks like an answer, use it
        if len(last) < 50:
            return _clean_answer(last)

    return None


def _clean_answer(raw: str) -> str:
    """Clean up an extracted answer string."""
    ans = raw.strip()
    # Remove trailing periods, commas
    ans = ans.rstrip('.,;')
    # Remove markdown formatting
    ans = re.sub(r'\*\*(.+?)\*\*', r'\1', ans)
    ans = re.sub(r'`(.+?)`', r'\1', ans)
    # Remove LaTeX formatting: \( ... \), \[ ... \], $ ... $, \boxed{...}
    ans = re.sub(r'\\\((.+?)\\\)', r'\1', ans)
    ans = re.sub(r'\\\[(.+?)\\\]', r'\1', ans)
    ans = re.sub(r'\\boxed\{(.+?)\}', r'\1', ans)
    ans = re.sub(r'\$(.+?)\$', r'\1', ans)
    # Remove LaTeX command artifacts
    ans = ans.replace('\\,', '').replace('\\;', '').replace('\\text{', '').rstrip('}')
    # Remove "x = " prefix (for algebra answers)
    ans = re.sub(r'^[a-zA-Z]\s*=\s*', '', ans)
    # Remove surrounding quotes
    ans = ans.strip('\'"')
    return ans.strip()


# ---------------------------------------------------------------------------
# Answer normalization & comparison
# ---------------------------------------------------------------------------

def normalize_answer(parsed: str, expected: str, tolerance: float = 1e-6) -> bool:
    """Compare a parsed answer against the expected answer.

    Handles:
        - Numeric comparison with floating-point tolerance
        - Boolean comparison (True/False/true/false)
        - String normalization (whitespace, case for booleans)
        - List/set comparison
    """
    if not parsed or not expected:
        return False

    p = parsed.strip()
    e = expected.strip()

    # Strip "x = " style prefixes from both sides
    p = re.sub(r'^[a-zA-Z]\s*=\s*', '', p)
    e = re.sub(r'^[a-zA-Z]\s*=\s*', '', e)

    # Direct string match (case-insensitive for booleans)
    if p.lower() == e.lower():
        return True

    # Boolean comparison
    bool_map = {'true': True, 'false': False, 'yes': True, 'no': False, '1': True, '0': False}
    if p.lower() in bool_map and e.lower() in bool_map:
        return bool_map[p.lower()] == bool_map[e.lower()]

    # Numeric comparison
    p_num = _try_parse_number(p)
    e_num = _try_parse_number(e)
    if p_num is not None and e_num is not None:
        if e_num == 0:
            return abs(p_num - e_num) < tolerance
        return abs(p_num - e_num) < tolerance or abs(p_num - e_num) / max(abs(e_num), 1) < tolerance

    # List comparison (for set operations, etc.)
    p_list = _try_parse_list(p)
    e_list = _try_parse_list(e)
    if p_list is not None and e_list is not None:
        return p_list == e_list

    # Exact string match after normalization
    return _normalize_str(p) == _normalize_str(e)


def _try_parse_number(s: str) -> float | None:
    """Try to parse a string as a number."""
    # Remove commas (thousands separators)
    s = s.replace(',', '').strip()
    # Remove dollar signs, percent signs
    s = s.strip('$%')
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    # Try evaluating simple arithmetic expressions safely via AST
    try:
        if re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', s):
            return float(_safe_eval_arithmetic(s))
    except Exception:
        pass
    return None


def _safe_eval_arithmetic(expr: str) -> float:
    """Safely evaluate a simple arithmetic expression using AST."""
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    tree = ast.parse(expr.strip(), mode='eval')
    return _eval(tree)


def _try_parse_list(s: str) -> list | None:
    """Try to parse a string as a list of numbers."""
    # Match patterns like [1, 2, 3] or {1, 2, 3} or 1, 2, 3
    s = s.strip()
    s = s.strip('[]{}()')
    if not s:
        return []
    try:
        items = [int(x.strip()) for x in s.split(',') if x.strip()]
        return sorted(items)
    except (ValueError, TypeError):
        pass
    return None


def _normalize_str(s: str) -> str:
    """Normalize a string for comparison."""
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    return s


# ---------------------------------------------------------------------------
# High-level verification
# ---------------------------------------------------------------------------

def verify_answer(response: str, expected: str, tolerance: float = 1e-6) -> VerificationResult:
    """Verify a model's full response against the expected answer.

    Args:
        response: The full model response (chain-of-thought + answer).
        expected: The correct answer string.
        tolerance: Numeric tolerance for floating-point comparison.

    Returns:
        VerificationResult with correctness, parsed answer, etc.
    """
    parsed = extract_answer(response)

    if parsed is None:
        return VerificationResult(
            is_correct=False,
            parsed_answer="[PARSE_FAILED]",
            expected_answer=expected,
            raw_response=response,
        )

    is_correct = normalize_answer(parsed, expected, tolerance)

    return VerificationResult(
        is_correct=is_correct,
        parsed_answer=parsed,
        expected_answer=expected,
        raw_response=response,
    )


def verify_with_function(
    response: str,
    verification_func,
    expected: str,
) -> VerificationResult:
    """Verify using the problem's custom verification function.

    Falls back to standard verify_answer if the function raises.
    """
    parsed = extract_answer(response)
    if parsed is None:
        parsed_display = "[PARSE_FAILED]"
        is_correct = False
    else:
        parsed_display = parsed
        try:
            is_correct = verification_func(response)
        except Exception:
            # Fallback to standard comparison
            is_correct = normalize_answer(parsed, expected)

    return VerificationResult(
        is_correct=is_correct,
        parsed_answer=parsed_display,
        expected_answer=expected,
        raw_response=response,
    )
