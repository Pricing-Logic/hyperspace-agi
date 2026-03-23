"""Problem Generator — template-based math/logic problems with ground-truth answers.

Generates problems across 5 domains and 5 difficulty tiers using randomized
templates. No LLM involved — every problem has a deterministic correct answer
and a verification function.
"""

import math
import random
import operator
import itertools
from dataclasses import dataclass
from typing import Callable


@dataclass
class Problem:
    text: str
    correct_answer: str
    domain: str
    difficulty: int
    verification_func: Callable[[str], bool]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _primes_below(n: int) -> list[int]:
    return [x for x in range(2, n) if _is_prime(x)]


def _fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (0-indexed: F(0)=0, F(1)=1, ...)."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _make_verifier(answer: str, tolerance: float = 1e-6) -> Callable[[str], bool]:
    """Build a verification function that accepts common answer formats."""
    def verify(response: str) -> bool:
        from src.loop.verifier import extract_answer, normalize_answer
        parsed = extract_answer(response)
        if parsed is None:
            parsed = response.strip()
        return normalize_answer(parsed, answer, tolerance=tolerance)
    return verify


# ---------------------------------------------------------------------------
# Domain: Arithmetic
# ---------------------------------------------------------------------------

def _arithmetic_tier1() -> Problem:
    ops = [("+", operator.add), ("-", operator.sub), ("*", operator.mul)]
    sym, fn = random.choice(ops)
    a = random.randint(2, 99)
    b = random.randint(2, 99)
    ans = fn(a, b)
    text = f"What is {a} {sym} {b}?"
    answer = str(ans)
    return Problem(text, answer, "arithmetic", 1, _make_verifier(answer))


def _arithmetic_tier2() -> Problem:
    """Two-step: a op1 b op2 c with standard left-to-right or order-of-ops."""
    templates = [
        lambda: _arith_order_of_ops(),
        lambda: _arith_two_step_word(),
    ]
    return random.choice(templates)()


def _arith_order_of_ops() -> Problem:
    a = random.randint(2, 20)
    b = random.randint(2, 20)
    c = random.randint(2, 20)
    # a + b * c  (tests order of operations)
    ans = a + b * c
    text = f"What is {a} + {b} * {c}? (Use standard order of operations.)"
    answer = str(ans)
    return Problem(text, answer, "arithmetic", 2, _make_verifier(answer))


def _arith_two_step_word() -> Problem:
    price = random.randint(5, 50)
    qty = random.randint(2, 10)
    discount = random.randint(1, price * qty - 1)
    total = price * qty - discount
    text = (
        f"A store sells widgets for ${price} each. You buy {qty} widgets "
        f"and get a ${discount} discount. How much do you pay in total?"
    )
    answer = str(total)
    return Problem(text, answer, "arithmetic", 2, _make_verifier(answer))


def _arithmetic_tier3() -> Problem:
    """Multi-step: sum of primes, factorial digits, etc."""
    templates = [_arith_prime_sum, _arith_modular, _arith_digit_sum]
    return random.choice(templates)()


def _arith_prime_sum() -> Problem:
    limit = random.choice([20, 30, 40, 50])
    primes = _primes_below(limit)
    ans = sum(primes)
    text = f"Find the sum of all prime numbers less than {limit}."
    answer = str(ans)
    return Problem(text, answer, "arithmetic", 3, _make_verifier(answer))


def _arith_modular() -> Problem:
    base = random.randint(2, 12)
    exp = random.randint(3, 8)
    mod = random.randint(3, 13)
    ans = pow(base, exp, mod)
    text = f"What is {base}^{exp} mod {mod}? (Compute {base} raised to the power {exp}, then take the remainder when divided by {mod}.)"
    answer = str(ans)
    return Problem(text, answer, "arithmetic", 3, _make_verifier(answer))


def _arith_digit_sum() -> Problem:
    n = random.randint(2, 8)
    fact = math.factorial(n)
    dsum = sum(int(d) for d in str(fact))
    text = f"What is the sum of the digits of {n}! ({n} factorial)?"
    answer = str(dsum)
    return Problem(text, answer, "arithmetic", 3, _make_verifier(answer))


def _arithmetic_tier4() -> Problem:
    """Complex arithmetic — big modular, nested, or multi-concept."""
    a = random.randint(100, 999)
    b = random.randint(100, 999)
    m = random.randint(7, 29)
    product = a * b
    ans = product % m
    text = f"What is ({a} * {b}) mod {m}?"
    answer = str(ans)
    return Problem(text, answer, "arithmetic", 4, _make_verifier(answer))


# ---------------------------------------------------------------------------
# Domain: Algebra
# ---------------------------------------------------------------------------

def _algebra_tier1() -> Problem:
    """Simple solve-for-x: x + a = b."""
    a = random.randint(1, 50)
    b = random.randint(a + 1, a + 50)
    ans = b - a
    text = f"Solve for x: x + {a} = {b}"
    answer = str(ans)
    return Problem(text, answer, "algebra", 1, _make_verifier(answer))


def _algebra_tier2() -> Problem:
    """Two-step: If x + a = b, what is x * c?"""
    a = random.randint(1, 20)
    b = random.randint(a + 1, a + 30)
    c = random.randint(2, 10)
    x = b - a
    ans = x * c
    text = f"If x + {a} = {b}, what is x * {c}?"
    answer = str(ans)
    return Problem(text, answer, "algebra", 2, _make_verifier(answer))


def _algebra_tier3() -> Problem:
    """Polynomial evaluation or system of equations."""
    templates = [_alg_poly_eval, _alg_system_2x2]
    return random.choice(templates)()


def _alg_poly_eval() -> Problem:
    a = random.randint(1, 5)
    b = random.randint(-10, 10)
    c = random.randint(-10, 10)
    x = random.randint(-5, 5)
    ans = a * x * x + b * x + c
    b_str = f"+ {b}" if b >= 0 else f"- {abs(b)}"
    c_str = f"+ {c}" if c >= 0 else f"- {abs(c)}"
    text = f"Evaluate f(x) = {a}x^2 {b_str}x {c_str} at x = {x}."
    answer = str(ans)
    return Problem(text, answer, "algebra", 3, _make_verifier(answer))


def _alg_system_2x2() -> Problem:
    """Generate a 2x2 system with integer solutions."""
    x_sol = random.randint(-10, 10)
    y_sol = random.randint(-10, 10)
    a1 = random.randint(1, 5)
    b1 = random.randint(1, 5)
    a2 = random.randint(1, 5)
    b2 = random.randint(1, 5)
    # Ensure the system is non-degenerate
    while a1 * b2 == a2 * b1:
        b2 = random.randint(1, 5)
    c1 = a1 * x_sol + b1 * y_sol
    c2 = a2 * x_sol + b2 * y_sol
    text = (
        f"Solve the system of equations:\n"
        f"  {a1}x + {b1}y = {c1}\n"
        f"  {a2}x + {b2}y = {c2}\n"
        f"What is x + y?"
    )
    answer = str(x_sol + y_sol)
    return Problem(text, answer, "algebra", 3, _make_verifier(answer))


def _algebra_tier4() -> Problem:
    """Quadratic — find sum/product of roots."""
    # ax^2 + bx + c = 0 where roots are r1, r2 (integers)
    r1 = random.randint(-8, 8)
    r2 = random.randint(-8, 8)
    a_coeff = 1
    b_coeff = -(r1 + r2)
    c_coeff = r1 * r2
    b_str = f"+ {b_coeff}" if b_coeff >= 0 else f"- {abs(b_coeff)}"
    c_str = f"+ {c_coeff}" if c_coeff >= 0 else f"- {abs(c_coeff)}"
    text = f"For the equation x^2 {b_str}x {c_str} = 0, what is the sum of its roots?"
    answer = str(r1 + r2)
    return Problem(text, answer, "algebra", 4, _make_verifier(answer))


# ---------------------------------------------------------------------------
# Domain: Logic
# ---------------------------------------------------------------------------

def _logic_tier1() -> Problem:
    """Simple boolean logic."""
    a = random.choice([True, False])
    b = random.choice([True, False])
    ops = [
        ("AND", a and b),
        ("OR", a or b),
    ]
    op_name, result = random.choice(ops)
    text = f"What is {a} {op_name} {b}? Answer True or False."
    answer = str(result)
    return Problem(text, answer, "logic", 1, _make_verifier(answer))


def _logic_tier2() -> Problem:
    """Set operations or compound boolean."""
    a = random.choice([True, False])
    b = random.choice([True, False])
    c = random.choice([True, False])
    result = (a and b) or c
    text = f"What is ({a} AND {b}) OR {c}? Answer True or False."
    answer = str(result)
    return Problem(text, answer, "logic", 2, _make_verifier(answer))


def _logic_tier3() -> Problem:
    """Counting / combinatorics or set cardinality."""
    templates = [_logic_counting, _logic_set_ops]
    return random.choice(templates)()


def _logic_counting() -> Problem:
    n = random.randint(4, 8)
    r = random.randint(2, min(4, n))
    ans = math.comb(n, r)
    text = f"How many ways can you choose {r} items from a set of {n}? (Compute C({n},{r}))"
    answer = str(ans)
    return Problem(text, answer, "logic", 3, _make_verifier(answer))


def _logic_set_ops() -> Problem:
    set_a = set(random.sample(range(1, 20), random.randint(3, 7)))
    set_b = set(random.sample(range(1, 20), random.randint(3, 7)))
    op = random.choice(["intersection", "union"])
    if op == "intersection":
        result = sorted(set_a & set_b)
        text = f"Find the intersection of {sorted(set_a)} and {sorted(set_b)}. List the elements in ascending order."
    else:
        result = sorted(set_a | set_b)
        text = f"Find the union of {sorted(set_a)} and {sorted(set_b)}. List the elements in ascending order."

    answer = str(result)

    def verify(response: str) -> bool:
        from src.loop.verifier import extract_answer
        parsed = extract_answer(response)
        if parsed is None:
            parsed = response.strip()
        # Try to extract numbers from the response
        import re
        nums = sorted(int(x) for x in re.findall(r'-?\d+', parsed))
        return nums == result

    return Problem(text, answer, "logic", 3, verify)


def _logic_tier4() -> Problem:
    """Permutation counting."""
    n = random.randint(4, 7)
    r = random.randint(2, min(4, n))
    ans = math.perm(n, r)
    text = (
        f"How many permutations of {r} items from a set of {n}? "
        f"(Compute P({n},{r}) = {n}! / ({n}-{r})!)"
    )
    answer = str(ans)
    return Problem(text, answer, "logic", 4, _make_verifier(answer))


# ---------------------------------------------------------------------------
# Domain: Sequences
# ---------------------------------------------------------------------------

def _seq_tier1() -> Problem:
    """Simple arithmetic sequence — find next term."""
    start = random.randint(1, 20)
    diff = random.randint(1, 10)
    seq = [start + i * diff for i in range(5)]
    ans = start + 5 * diff
    text = f"What is the next number in the sequence: {', '.join(map(str, seq))}, ...?"
    answer = str(ans)
    return Problem(text, answer, "sequences", 1, _make_verifier(answer))


def _seq_tier2() -> Problem:
    """Geometric sequence — find next term."""
    start = random.randint(1, 5)
    ratio = random.randint(2, 4)
    seq = [start * ratio**i for i in range(5)]
    ans = start * ratio**5
    text = f"What is the next number in the sequence: {', '.join(map(str, seq))}, ...?"
    answer = str(ans)
    return Problem(text, answer, "sequences", 2, _make_verifier(answer))


def _seq_tier3() -> Problem:
    """Fibonacci-style or sum-of-previous."""
    templates = [_seq_fib_nth, _seq_sum_arith]
    return random.choice(templates)()


def _seq_fib_nth() -> Problem:
    n = random.randint(6, 12)
    ans = _fibonacci(n)
    text = (
        f"In the Fibonacci sequence (starting 0, 1, 1, 2, 3, 5, ...), "
        f"what is the {n}th term? (Use 0-based indexing: F(0)=0, F(1)=1, F(2)=1, ...)"
    )
    answer = str(ans)
    return Problem(text, answer, "sequences", 3, _make_verifier(answer))


def _seq_sum_arith() -> Problem:
    n = random.randint(10, 50)
    ans = n * (n + 1) // 2
    text = f"What is the sum of all integers from 1 to {n}?"
    answer = str(ans)
    return Problem(text, answer, "sequences", 3, _make_verifier(answer))


def _seq_tier4() -> Problem:
    """Fibonacci mod or triangular numbers."""
    n = random.randint(8, 15)
    m = random.randint(3, 11)
    fib_val = _fibonacci(n)
    ans = fib_val % m
    text = f"What is the {n}th Fibonacci number modulo {m}? (F(0)=0, F(1)=1, 0-indexed)"
    answer = str(ans)
    return Problem(text, answer, "sequences", 4, _make_verifier(answer))


# ---------------------------------------------------------------------------
# Domain: Code Reasoning
# ---------------------------------------------------------------------------

def _code_tier1() -> Problem:
    """What does this simple code output?"""
    a = random.randint(1, 20)
    b = random.randint(1, 20)
    ans = a + b
    code = f"x = {a}\ny = {b}\nprint(x + y)"
    text = f"What does this Python code output?\n\n```python\n{code}\n```"
    answer = str(ans)
    return Problem(text, answer, "code_reasoning", 1, _make_verifier(answer))


def _code_tier2() -> Problem:
    """Simple loop."""
    n = random.randint(3, 8)
    ans = sum(range(n))
    code = (
        f"total = 0\n"
        f"for i in range({n}):\n"
        f"    total += i\n"
        f"print(total)"
    )
    text = f"What does this Python code output?\n\n```python\n{code}\n```"
    answer = str(ans)
    return Problem(text, answer, "code_reasoning", 2, _make_verifier(answer))


def _code_tier3() -> Problem:
    """List comprehension or conditional."""
    templates = [_code_list_comp, _code_conditional_loop]
    return random.choice(templates)()


def _code_list_comp() -> Problem:
    limit = random.randint(5, 15)
    ans = [x**2 for x in range(1, limit + 1) if x % 2 == 0]
    code = f"result = [x**2 for x in range(1, {limit + 1}) if x % 2 == 0]\nprint(sum(result))"
    answer = str(sum(ans))
    text = f"What does this Python code output?\n\n```python\n{code}\n```"
    return Problem(text, answer, "code_reasoning", 3, _make_verifier(answer))


def _code_conditional_loop() -> Problem:
    n = random.randint(10, 30)
    count = sum(1 for x in range(1, n + 1) if x % 3 == 0 or x % 5 == 0)
    code = (
        f"count = 0\n"
        f"for i in range(1, {n + 1}):\n"
        f"    if i % 3 == 0 or i % 5 == 0:\n"
        f"        count += 1\n"
        f"print(count)"
    )
    text = f"What does this Python code output?\n\n```python\n{code}\n```"
    answer = str(count)
    return Problem(text, answer, "code_reasoning", 3, _make_verifier(answer))


def _code_tier4() -> Problem:
    """Recursive or dictionary-based code."""
    n = random.randint(3, 7)
    # Factorial via recursion
    ans = math.factorial(n)
    code = (
        f"def factorial(n):\n"
        f"    if n <= 1:\n"
        f"        return 1\n"
        f"    return n * factorial(n - 1)\n"
        f"\n"
        f"print(factorial({n}))"
    )
    text = f"What does this Python code output?\n\n```python\n{code}\n```"
    answer = str(ans)
    return Problem(text, answer, "code_reasoning", 4, _make_verifier(answer))


# ---------------------------------------------------------------------------
# Tier 5: Cross-domain compositional problems
# ---------------------------------------------------------------------------

def _tier5_composite() -> Problem:
    """Combines multiple domains into one problem."""
    templates = [
        _composite_fib_prime,
        _composite_code_algebra,
        _composite_seq_modular,
    ]
    return random.choice(templates)()


def _composite_fib_prime() -> Problem:
    """How many Fibonacci numbers below N are prime?"""
    limit = random.choice([50, 100, 200])
    fibs = []
    a, b = 0, 1
    while a < limit:
        fibs.append(a)
        a, b = b, a + b
    prime_fibs = [f for f in fibs if _is_prime(f)]
    ans = len(prime_fibs)
    text = f"How many Fibonacci numbers less than {limit} are also prime?"
    answer = str(ans)
    return Problem(text, answer, "composite", 5, _make_verifier(answer))


def _composite_code_algebra() -> Problem:
    """Code that solves a quadratic — what does it output?"""
    # x^2 - (r1+r2)x + r1*r2 = 0
    r1 = random.randint(1, 10)
    r2 = random.randint(1, 10)
    while r1 == r2:
        r2 = random.randint(1, 10)
    b = -(r1 + r2)
    c = r1 * r2
    # Code finds the smaller root
    ans = min(r1, r2)
    code = (
        f"import math\n"
        f"a, b, c = 1, {b}, {c}\n"
        f"discriminant = b**2 - 4*a*c\n"
        f"root1 = (-b + math.sqrt(discriminant)) / (2*a)\n"
        f"root2 = (-b - math.sqrt(discriminant)) / (2*a)\n"
        f"print(int(min(root1, root2)))"
    )
    text = f"What does this Python code output?\n\n```python\n{code}\n```"
    answer = str(ans)
    return Problem(text, answer, "composite", 5, _make_verifier(answer))


def _composite_seq_modular() -> Problem:
    """Sum of first N terms of an arithmetic sequence, modulo M."""
    start = random.randint(1, 10)
    diff = random.randint(1, 5)
    n = random.randint(5, 15)
    m = random.randint(3, 13)
    total = sum(start + i * diff for i in range(n))
    ans = total % m
    text = (
        f"Consider the arithmetic sequence starting at {start} with common difference {diff}. "
        f"What is the sum of the first {n} terms, modulo {m}?"
    )
    answer = str(ans)
    return Problem(text, answer, "composite", 5, _make_verifier(answer))


# ---------------------------------------------------------------------------
# Generator dispatch table
# ---------------------------------------------------------------------------

_GENERATORS: dict[str, dict[int, Callable[[], Problem]]] = {
    "arithmetic": {1: _arithmetic_tier1, 2: _arithmetic_tier2, 3: _arithmetic_tier3, 4: _arithmetic_tier4},
    "algebra":    {1: _algebra_tier1,    2: _algebra_tier2,    3: _algebra_tier3,    4: _algebra_tier4},
    "logic":      {1: _logic_tier1,      2: _logic_tier2,      3: _logic_tier3,      4: _logic_tier4},
    "sequences":  {1: _seq_tier1,        2: _seq_tier2,        3: _seq_tier3,        4: _seq_tier4},
    "code_reasoning": {1: _code_tier1,   2: _code_tier2,       3: _code_tier3,       4: _code_tier4},
}

DOMAINS = list(_GENERATORS.keys())


def generate_problem(difficulty: int = 1, domain: str | None = None) -> Problem:
    """Generate a single problem at the given difficulty.

    Args:
        difficulty: 1-5 (tier 5 is always cross-domain composite).
        domain: A specific domain, or None to pick randomly.

    Returns:
        A Problem dataclass with text, answer, verifier, etc.
    """
    difficulty = max(1, min(5, difficulty))

    if difficulty == 5:
        return _tier5_composite()

    if domain is None:
        domain = random.choice(DOMAINS)

    if domain not in _GENERATORS:
        domain = random.choice(DOMAINS)

    # Clamp difficulty to available tiers for this domain
    max_tier = max(_GENERATORS[domain].keys())
    tier = min(difficulty, max_tier)
    return _GENERATORS[domain][tier]()


def generate_batch(
    n: int,
    difficulty: int = 1,
    domain: str | None = None,
    domain_distribution: dict[str, float] | None = None,
) -> list[Problem]:
    """Generate a batch of n problems.

    Args:
        n: Number of problems.
        difficulty: Target difficulty tier.
        domain: Fixed domain, or None for mixed.
        domain_distribution: Optional weights like {"arithmetic": 0.3, "logic": 0.3, ...}.

    Returns:
        List of Problem instances.
    """
    problems = []
    for _ in range(n):
        if domain_distribution:
            d = random.choices(
                list(domain_distribution.keys()),
                weights=list(domain_distribution.values()),
                k=1,
            )[0]
            problems.append(generate_problem(difficulty, d))
        else:
            problems.append(generate_problem(difficulty, domain))
    return problems


# ---------------------------------------------------------------------------
# Curriculum manager
# ---------------------------------------------------------------------------

class CurriculumManager:
    """Tracks per-domain success rates and manages difficulty progression."""

    def __init__(self, start_difficulty: int = 1, advance_threshold: float = 0.80,
                 retreat_threshold: float = 0.30):
        self.difficulty = start_difficulty
        self.advance_threshold = advance_threshold
        self.retreat_threshold = retreat_threshold
        self.domain_stats: dict[str, dict] = {
            d: {"correct": 0, "total": 0} for d in DOMAINS + ["composite"]
        }
        self.history: list[dict] = []

    def record(self, domain: str, correct: bool) -> None:
        if domain not in self.domain_stats:
            self.domain_stats[domain] = {"correct": 0, "total": 0}
        self.domain_stats[domain]["total"] += 1
        if correct:
            self.domain_stats[domain]["correct"] += 1

    def overall_accuracy(self) -> float:
        total = sum(s["total"] for s in self.domain_stats.values())
        correct = sum(s["correct"] for s in self.domain_stats.values())
        return correct / total if total > 0 else 0.0

    def domain_accuracy(self, domain: str) -> float:
        s = self.domain_stats.get(domain, {"correct": 0, "total": 0})
        return s["correct"] / s["total"] if s["total"] > 0 else 0.0

    def update_difficulty(self) -> tuple[int, int]:
        """Check accuracy and adjust difficulty. Returns (old, new) difficulty."""
        old = self.difficulty
        acc = self.overall_accuracy()
        if acc >= self.advance_threshold and self.difficulty < 5:
            self.difficulty += 1
        elif acc <= self.retreat_threshold and self.difficulty > 1:
            self.difficulty -= 1

        self.history.append({
            "old_difficulty": old,
            "new_difficulty": self.difficulty,
            "accuracy": acc,
        })
        return old, self.difficulty

    def reset_generation_stats(self) -> None:
        """Reset per-generation stats (call between generations)."""
        for d in self.domain_stats:
            self.domain_stats[d] = {"correct": 0, "total": 0}

    def get_domain_report(self) -> dict[str, float]:
        return {d: self.domain_accuracy(d) for d in self.domain_stats if self.domain_stats[d]["total"] > 0}
