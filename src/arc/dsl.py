"""Growing DSL (Domain-Specific Language) of grid primitives for ARC puzzles.

Starts with ~20 hand-coded primitives. The DSLLibrary can grow at runtime
when the system discovers new useful transformations.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Primitive implementations
# ---------------------------------------------------------------------------

def rotate_90(grid: np.ndarray) -> np.ndarray:
    """Rotate the grid 90 degrees clockwise."""
    return np.rot90(grid, k=-1)


def rotate_180(grid: np.ndarray) -> np.ndarray:
    """Rotate the grid 180 degrees."""
    return np.rot90(grid, k=2)


def rotate_270(grid: np.ndarray) -> np.ndarray:
    """Rotate the grid 270 degrees clockwise (= 90 counter-clockwise)."""
    return np.rot90(grid, k=1)


def flip_horizontal(grid: np.ndarray) -> np.ndarray:
    """Flip the grid left-to-right."""
    return np.fliplr(grid)


def flip_vertical(grid: np.ndarray) -> np.ndarray:
    """Flip the grid top-to-bottom."""
    return np.flipud(grid)


def transpose(grid: np.ndarray) -> np.ndarray:
    """Transpose the grid (swap rows and columns)."""
    return grid.T.copy()


def crop_to_content(grid: np.ndarray, background: int = 0) -> np.ndarray:
    """Remove border rows/columns that are entirely the background color.

    Args:
        grid: Input grid.
        background: Color value to treat as background (default 0).

    Returns:
        Cropped grid with background-only borders removed.
    """
    mask = grid != background
    if not mask.any():
        return grid.copy()

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return grid[r_min:r_max + 1, c_min:c_max + 1].copy()


def flood_fill(grid: np.ndarray, row: int, col: int, new_color: int) -> np.ndarray:
    """Flood-fill from (row, col) with new_color using 4-connectivity.

    Args:
        grid: Input grid.
        row: Starting row.
        col: Starting column.
        new_color: Color to fill with.

    Returns:
        New grid with the fill applied.
    """
    result = grid.copy()
    h, w = result.shape
    old_color = int(result[row, col])
    if old_color == new_color:
        return result

    queue = deque([(row, col)])
    result[row, col] = new_color

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and result[nr, nc] == old_color:
                result[nr, nc] = new_color
                queue.append((nr, nc))

    return result


def find_objects(grid: np.ndarray, background: int = 0) -> list[np.ndarray]:
    """Find connected components (objects) of non-background color.

    Uses 4-connectivity. Each returned array is cropped to the bounding
    box of the component, with background color elsewhere.

    Args:
        grid: Input grid.
        background: Background color to ignore.

    Returns:
        List of cropped grids, one per connected component.
    """
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or grid[r, c] == background:
                continue

            cells = []
            color = int(grid[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True

            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            obj = np.full((max_r - min_r + 1, max_c - min_c + 1), background, dtype=int)
            for cr, cc in cells:
                obj[cr - min_r, cc - min_c] = grid[cr, cc]
            objects.append(obj)

    return objects


def get_colors(grid: np.ndarray) -> set[int]:
    """Return the set of unique color values in the grid."""
    return set(int(v) for v in np.unique(grid))


def count_colors(grid: np.ndarray) -> dict[int, int]:
    """Return a dict mapping each color to its pixel count."""
    values, counts = np.unique(grid, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def scale_up(grid: np.ndarray, factor: int) -> np.ndarray:
    """Enlarge the grid by an integer factor (each cell becomes factor x factor).

    Args:
        grid: Input grid.
        factor: Scale factor (must be >= 1).

    Returns:
        Scaled grid.
    """
    return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)


def tile(grid: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Tile (repeat) the grid in a rows x cols arrangement.

    Args:
        grid: Input grid.
        rows: Number of vertical repeats.
        cols: Number of horizontal repeats.

    Returns:
        Tiled grid.
    """
    return np.tile(grid, (rows, cols))


def extract_subgrid(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
    """Extract a rectangular subgrid from (r1,c1) to (r2,c2) inclusive.

    Args:
        grid: Input grid.
        r1, c1: Top-left corner (inclusive).
        r2, c2: Bottom-right corner (inclusive).

    Returns:
        Extracted subgrid.
    """
    return grid[r1:r2 + 1, c1:c2 + 1].copy()


def replace_color(grid: np.ndarray, old_color: int, new_color: int) -> np.ndarray:
    """Replace all pixels of old_color with new_color.

    Args:
        grid: Input grid.
        old_color: Color to replace.
        new_color: Replacement color.

    Returns:
        New grid with colors replaced.
    """
    result = grid.copy()
    result[result == old_color] = new_color
    return result


def overlay(base: np.ndarray, top: np.ndarray, transparent: int = 0) -> np.ndarray:
    """Overlay 'top' grid onto 'base' grid. Pixels in 'top' with the
    transparent color are see-through (base shows through).

    Both grids must have the same shape.

    Args:
        base: Background grid.
        top: Foreground grid.
        transparent: Color value treated as transparent in the top grid.

    Returns:
        Combined grid.
    """
    if base.shape != top.shape:
        # If sizes differ, place top at (0,0) on a copy of base
        result = base.copy()
        h = min(base.shape[0], top.shape[0])
        w = min(base.shape[1], top.shape[1])
        mask = top[:h, :w] != transparent
        result[:h, :w][mask] = top[:h, :w][mask]
        return result

    result = base.copy()
    mask = top != transparent
    result[mask] = top[mask]
    return result


def border(grid: np.ndarray, color: int, width: int = 1) -> np.ndarray:
    """Add a border of the given color around the grid.

    Args:
        grid: Input grid.
        color: Border color.
        width: Border width in cells.

    Returns:
        Grid with border added.
    """
    h, w = grid.shape
    new_h, new_w = h + 2 * width, w + 2 * width
    result = np.full((new_h, new_w), color, dtype=int)
    result[width:width + h, width:width + w] = grid
    return result


def gravity_down(grid: np.ndarray, background: int = 0) -> np.ndarray:
    """Apply gravity: non-background cells fall to the bottom of each column.

    Args:
        grid: Input grid.
        background: Background color.

    Returns:
        Grid with gravity applied column-wise.
    """
    result = np.full_like(grid, background)
    h, w = grid.shape
    for c in range(w):
        col = grid[:, c]
        non_bg = col[col != background]
        if len(non_bg) > 0:
            result[h - len(non_bg):h, c] = non_bg
    return result


def mirror_diagonal(grid: np.ndarray) -> np.ndarray:
    """Mirror along the main diagonal (top-left to bottom-right).

    For non-square grids this is the same as transpose.
    """
    return grid.T.copy()


def count_nonzero(grid: np.ndarray) -> int:
    """Count non-zero cells in the grid."""
    return int(np.count_nonzero(grid))


def most_common_color(grid: np.ndarray, exclude_background: bool = True) -> int:
    """Return the most common color in the grid.

    Args:
        grid: Input grid.
        exclude_background: If True, exclude color 0.

    Returns:
        The most frequent color value.
    """
    values, counts = np.unique(grid, return_counts=True)
    color_counts = dict(zip(values, counts))
    if exclude_background and 0 in color_counts and len(color_counts) > 1:
        del color_counts[0]
    return int(max(color_counts, key=color_counts.get))


def pad_grid(grid: np.ndarray, target_h: int, target_w: int, fill: int = 0) -> np.ndarray:
    """Pad grid to target dimensions, filling with fill color.

    Args:
        grid: Input grid.
        target_h: Target height.
        target_w: Target width.
        fill: Fill color for padding.

    Returns:
        Padded grid.
    """
    result = np.full((target_h, target_w), fill, dtype=int)
    h = min(grid.shape[0], target_h)
    w = min(grid.shape[1], target_w)
    result[:h, :w] = grid[:h, :w]
    return result


# ---------------------------------------------------------------------------
# DSL Library — stores primitives and tracks usage
# ---------------------------------------------------------------------------

@dataclass
class PrimitiveInfo:
    """Metadata about a single DSL primitive."""
    name: str
    func: Callable
    docstring: str
    signature: str
    usage_count: int = 0


class DSLLibrary:
    """Registry of grid transformation primitives.

    Stores built-in and discovered primitives, tracks usage,
    and generates prompt-friendly descriptions for the LLM.
    """

    def __init__(self):
        self._primitives: dict[str, PrimitiveInfo] = {}
        self._register_builtins()

    def _register_builtins(self):
        """Register all built-in primitives."""
        builtins = [
            (rotate_90, "rotate_90(grid) -> np.ndarray"),
            (rotate_180, "rotate_180(grid) -> np.ndarray"),
            (rotate_270, "rotate_270(grid) -> np.ndarray"),
            (flip_horizontal, "flip_horizontal(grid) -> np.ndarray"),
            (flip_vertical, "flip_vertical(grid) -> np.ndarray"),
            (transpose, "transpose(grid) -> np.ndarray"),
            (crop_to_content, "crop_to_content(grid, background=0) -> np.ndarray"),
            (flood_fill, "flood_fill(grid, row, col, new_color) -> np.ndarray"),
            (find_objects, "find_objects(grid, background=0) -> list[np.ndarray]"),
            (get_colors, "get_colors(grid) -> set[int]"),
            (count_colors, "count_colors(grid) -> dict[int, int]"),
            (scale_up, "scale_up(grid, factor) -> np.ndarray"),
            (tile, "tile(grid, rows, cols) -> np.ndarray"),
            (extract_subgrid, "extract_subgrid(grid, r1, c1, r2, c2) -> np.ndarray"),
            (replace_color, "replace_color(grid, old_color, new_color) -> np.ndarray"),
            (overlay, "overlay(base, top, transparent=0) -> np.ndarray"),
            (border, "border(grid, color, width=1) -> np.ndarray"),
            (gravity_down, "gravity_down(grid, background=0) -> np.ndarray"),
            (mirror_diagonal, "mirror_diagonal(grid) -> np.ndarray"),
            (count_nonzero, "count_nonzero(grid) -> int"),
            (most_common_color, "most_common_color(grid, exclude_background=True) -> int"),
            (pad_grid, "pad_grid(grid, target_h, target_w, fill=0) -> np.ndarray"),
        ]
        for func, sig in builtins:
            self.register(func.__name__, func, sig)

    def register(self, name: str, func: Callable, signature: str | None = None):
        """Register a new primitive.

        Args:
            name: Name for the primitive.
            func: The function.
            signature: Human-readable signature string. Auto-generated if None.
        """
        doc = (func.__doc__ or "No description.").strip().split("\n")[0]
        if signature is None:
            signature = f"{name}(...)"
        self._primitives[name] = PrimitiveInfo(
            name=name,
            func=func,
            docstring=doc,
            signature=signature,
        )

    def get(self, name: str) -> Callable | None:
        """Get a primitive function by name."""
        info = self._primitives.get(name)
        return info.func if info else None

    def record_usage(self, name: str):
        """Record that a primitive was used (for tracking popularity)."""
        if name in self._primitives:
            self._primitives[name].usage_count += 1

    def get_all_functions(self) -> dict[str, Callable]:
        """Return dict mapping name -> function for all primitives."""
        return {name: info.func for name, info in self._primitives.items()}

    def get_prompt_description(self) -> str:
        """Generate a prompt-friendly description of all available primitives.

        This is injected into the LLM prompt so it knows what tools are available.
        """
        lines = ["Available grid primitives (all operate on np.ndarray grids with int values 0-9):"]
        lines.append("")

        for name, info in self._primitives.items():
            lines.append(f"  {info.signature}")
            lines.append(f"    {info.docstring}")
            lines.append("")

        return "\n".join(lines)

    def get_usage_stats(self) -> list[tuple[str, int]]:
        """Return primitives sorted by usage count (descending)."""
        return sorted(
            [(info.name, info.usage_count) for info in self._primitives.values()],
            key=lambda x: -x[1],
        )

    @property
    def names(self) -> list[str]:
        """List all primitive names."""
        return list(self._primitives.keys())

    def __len__(self) -> int:
        return len(self._primitives)

    def __contains__(self, name: str) -> bool:
        return name in self._primitives
