"""Safe program execution sandbox for ARC solver.

Runs generated Python programs in a restricted environment with:
- Timeout enforcement via multiprocessing
- Injected numpy + DSL primitives
- Blocked dangerous operations (file I/O, network, dangerous imports)
- AST-based validation of generated code
"""

import ast
import multiprocessing
import os
import sys
import traceback
from typing import Any

import numpy as np

from src.config import SANDBOX_TIMEOUT_SECONDS, MAX_PROGRAM_LENGTH
from src.arc.dsl import DSLLibrary


# Modules that generated code is allowed to import
_ALLOWED_MODULES = frozenset({
    "numpy", "np", "math", "itertools", "functools",
    "collections", "copy",
})

# Built-in functions that are blocked
_BLOCKED_BUILTINS = frozenset({
    "open", "exec", "eval", "compile", "__import__",
    "breakpoint", "exit", "quit",
    "input", "help", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "dir",
    "type", "super",
})

# Attribute names that are never allowed
_BLOCKED_ATTRS = frozenset({
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__code__", "__closure__", "__func__",
    "__self__", "__dict__", "__init_subclass__",
    "__import__", "__builtins__", "__loader__", "__spec__",
})


def _validate_ast(code: str) -> str | None:
    """Validate generated code via AST analysis. Returns error message or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"

    for node in ast.walk(tree):
        # Block access to dunder attributes
        if isinstance(node, ast.Attribute):
            if node.attr in _BLOCKED_ATTRS:
                return f"Blocked attribute access: {node.attr}"
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return f"Dunder attribute access not allowed: {node.attr}"

        # Block dangerous imports beyond our whitelist
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _ALLOWED_MODULES:
                    return f"Import of '{alias.name}' not allowed"

        if isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in _ALLOWED_MODULES:
                    return f"Import from '{node.module}' not allowed"

    return None


def _build_safe_builtins() -> dict:
    """Build a dict of safe built-in functions (blocking dangerous ones)."""
    import builtins
    safe = {}
    for name in dir(builtins):
        if name in _BLOCKED_BUILTINS:
            continue
        safe[name] = getattr(builtins, name)

    # Replace __import__ with a restricted version using importlib
    def safe_import(name, *args, **kwargs):
        top_level = name.split(".")[0]
        if top_level not in _ALLOWED_MODULES:
            raise ImportError(
                f"Import of '{name}' is not allowed. "
                f"Allowed modules: {', '.join(sorted(_ALLOWED_MODULES))}"
            )
        import importlib
        return importlib.import_module(name)

    safe["__import__"] = safe_import

    # Provide a safe getattr that blocks dunder access
    def safe_getattr(obj, name, *default):
        if isinstance(name, str) and name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"Access to '{name}' is not allowed")
        if isinstance(name, str) and name in _BLOCKED_ATTRS:
            raise AttributeError(f"Access to '{name}' is not allowed")
        return builtins.getattr(obj, name, *default)

    safe["getattr"] = safe_getattr

    return safe


def _build_execution_namespace(input_grid: np.ndarray, dsl: DSLLibrary | None = None) -> dict:
    """Build the namespace in which generated programs execute."""
    namespace = {"__builtins__": _build_safe_builtins()}

    # Inject numpy
    namespace["np"] = np
    namespace["numpy"] = np

    # Inject standard safe modules
    import math
    import itertools
    import functools
    import collections
    import copy
    namespace["math"] = math
    namespace["itertools"] = itertools
    namespace["functools"] = functools
    namespace["collections"] = collections
    namespace["copy"] = copy

    # Inject DSL primitives
    if dsl is not None:
        for name, func in dsl.get_all_functions().items():
            namespace[name] = func

    # Inject the input grid
    namespace["input_grid"] = input_grid.copy()
    namespace["grid"] = input_grid.copy()

    return namespace


def _execute_in_process(code: str, input_grid_list: list, project_root: str, result_queue: multiprocessing.Queue):
    """Worker function that runs in a separate process.

    Args:
        code: Python source code to execute.
        input_grid_list: Input grid as nested list (picklable).
        project_root: Project root path to add to sys.path.
        result_queue: Queue to put (result, error) tuple.
    """
    try:
        # Fix sys.path for macOS spawn mode (C3 fix)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        input_grid = np.array(input_grid_list, dtype=int)

        # Rebuild namespace inside the subprocess
        from src.arc.dsl import DSLLibrary
        dsl = DSLLibrary()
        namespace = _build_execution_namespace(input_grid, dsl)

        # Execute the generated code
        exec(code, namespace)

        # Look for the transform function
        if "transform" in namespace:
            result = namespace["transform"](input_grid)
            if isinstance(result, np.ndarray):
                result_queue.put((result.tolist(), None))
            else:
                result_queue.put((None, f"transform() returned {type(result).__name__}, expected np.ndarray"))
        else:
            result_queue.put((None, "No 'transform' function defined in generated code"))

    except Exception as e:
        tb = traceback.format_exc()
        if len(tb) > 1000:
            tb = tb[:500] + "\n...[truncated]...\n" + tb[-500:]
        result_queue.put((None, f"{type(e).__name__}: {e}\n{tb}"))


def execute_program(
    code: str,
    input_grid: np.ndarray,
    dsl: DSLLibrary | None = None,
    timeout: int | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """Execute a generated program safely in a sandboxed subprocess.

    Args:
        code: Python source code defining a transform(grid) function.
        input_grid: The input grid to transform.
        dsl: DSLLibrary instance (primitives are injected into the namespace).
        timeout: Timeout in seconds. Defaults to SANDBOX_TIMEOUT_SECONDS.

    Returns:
        (result_grid, error_message) — one will be None.
    """
    if timeout is None:
        timeout = SANDBOX_TIMEOUT_SECONDS

    # Enforce max program length
    if len(code) > MAX_PROGRAM_LENGTH:
        return None, f"Program too long ({len(code)} chars, max {MAX_PROGRAM_LENGTH})"

    # AST-based validation (catches dunder access, blocked imports, etc.)
    ast_error = _validate_ast(code)
    if ast_error is not None:
        return None, f"Code validation failed: {ast_error}"

    # Get project root for subprocess sys.path
    project_root = str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_execute_in_process,
        args=(code, input_grid.tolist(), project_root, result_queue),
    )
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        return None, f"Execution timed out after {timeout} seconds"

    if process.exitcode != 0 and result_queue.empty():
        return None, f"Process crashed with exit code {process.exitcode}"

    if result_queue.empty():
        return None, "No result returned from execution"

    result_list, error = result_queue.get_nowait()
    if error is not None:
        return None, error

    return np.array(result_list, dtype=int), None


def execute_program_inline(
    code: str,
    input_grid: np.ndarray,
    dsl: DSLLibrary | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """Execute a program in the current process (no subprocess isolation).

    Faster but less safe. Useful for trusted code or testing.
    """
    # Still validate AST even inline
    ast_error = _validate_ast(code)
    if ast_error is not None:
        return None, f"Code validation failed: {ast_error}"

    try:
        namespace = _build_execution_namespace(input_grid, dsl)
        exec(code, namespace)

        if "transform" not in namespace:
            return None, "No 'transform' function defined in generated code"

        result = namespace["transform"](input_grid)
        if not isinstance(result, np.ndarray):
            return None, f"transform() returned {type(result).__name__}, expected np.ndarray"

        return result.astype(int), None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
