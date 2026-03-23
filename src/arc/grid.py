"""Grid representation for ARC puzzles — wraps numpy arrays with display and analysis."""

from collections import deque

import numpy as np

# ARC color palette — maps 0-9 to ANSI 256-color codes for terminal display.
# 0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=grey, 6=magenta, 7=orange, 8=cyan, 9=maroon
_COLOR_ANSI = {
    0: "\033[48;5;0m",    # black
    1: "\033[48;5;21m",   # blue
    2: "\033[48;5;196m",  # red
    3: "\033[48;5;46m",   # green
    4: "\033[48;5;226m",  # yellow
    5: "\033[48;5;244m",  # grey
    6: "\033[48;5;201m",  # magenta
    7: "\033[48;5;208m",  # orange
    8: "\033[48;5;51m",   # cyan
    9: "\033[48;5;124m",  # maroon
}
_RESET = "\033[0m"
_BLOCK = "  "  # two spaces per cell for squarish appearance


class Grid:
    """Wraps a 2D numpy array (values 0-9) representing an ARC grid."""

    def __init__(self, data: np.ndarray | list):
        if isinstance(data, list):
            data = np.array(data, dtype=int)
        if data.ndim != 2:
            raise ValueError(f"Grid must be 2D, got shape {data.shape}")
        self.data = data.astype(int)

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def unique_colors(self) -> set[int]:
        """Return set of unique color values in this grid."""
        return set(int(v) for v in np.unique(self.data))

    def count_colors(self) -> dict[int, int]:
        """Return dict mapping each color to its pixel count."""
        values, counts = np.unique(self.data, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    def equals(self, other: "Grid") -> bool:
        """Check exact equality with another grid."""
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self.data, other.data)

    def pretty(self, use_color: bool = True) -> str:
        """Pretty-print the grid with colored unicode blocks.

        Args:
            use_color: If True, use ANSI colors. If False, show digit values.

        Returns:
            Multi-line string representation.
        """
        lines = []
        for row in self.data:
            if use_color:
                cells = []
                for val in row:
                    color = _COLOR_ANSI.get(int(val), _COLOR_ANSI[0])
                    cells.append(f"{color}{_BLOCK}{_RESET}")
                lines.append("".join(cells))
            else:
                lines.append(" ".join(str(int(v)) for v in row))
        return "\n".join(lines)

    def compact(self) -> str:
        """Compact text representation — digits only, one row per line.

        This is what we feed to the LLM in prompts (no ANSI codes).
        """
        return "\n".join(
            "".join(str(int(v)) for v in row)
            for row in self.data
        )

    def find_objects(self, background: int = 0) -> list["Grid"]:
        """Find connected components (objects) of non-background color.

        Uses 4-connectivity (up/down/left/right).

        Args:
            background: The background color to ignore (default 0).

        Returns:
            List of Grid objects, each containing one connected component
            (cropped to its bounding box, with background elsewhere).
        """
        visited = np.zeros_like(self.data, dtype=bool)
        objects = []

        for r in range(self.height):
            for c in range(self.width):
                if visited[r, c] or self.data[r, c] == background:
                    continue

                # BFS to find connected component
                component_cells = []
                color = int(self.data[r, c])
                queue = deque([(r, c)])
                visited[r, c] = True

                while queue:
                    cr, cc = queue.popleft()
                    component_cells.append((cr, cc))

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < self.height
                            and 0 <= nc < self.width
                            and not visited[nr, nc]
                            and self.data[nr, nc] == color
                        ):
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                # Extract bounding box
                rows = [cell[0] for cell in component_cells]
                cols = [cell[1] for cell in component_cells]
                min_r, max_r = min(rows), max(rows)
                min_c, max_c = min(cols), max(cols)

                obj = np.full((max_r - min_r + 1, max_c - min_c + 1), background, dtype=int)
                for cr, cc in component_cells:
                    obj[cr - min_r, cc - min_c] = self.data[cr, cc]
                objects.append(Grid(obj))

        return objects

    def __repr__(self) -> str:
        return f"Grid({self.height}x{self.width}, colors={self.unique_colors()})"

    def __eq__(self, other) -> bool:
        return self.equals(other)

    def __hash__(self):
        return hash(self.data.tobytes())


def grids_match(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if two numpy arrays represent the same grid."""
    if a.shape != b.shape:
        return False
    return np.array_equal(a, b)


def display_pair(input_grid: np.ndarray, output_grid: np.ndarray, pair_label: str = "Pair") -> str:
    """Format an input/output pair side by side for display."""
    g_in = Grid(input_grid)
    g_out = Grid(output_grid)
    lines = [f"--- {pair_label} ---"]
    lines.append(f"Input ({g_in.height}x{g_in.width}):")
    lines.append(g_in.pretty())
    lines.append(f"Output ({g_out.height}x{g_out.width}):")
    lines.append(g_out.pretty())
    lines.append("")
    return "\n".join(lines)


def format_grid_for_prompt(grid: np.ndarray) -> str:
    """Format a grid as a compact text string suitable for LLM prompts.

    Each row is a line of digits. This is the canonical format the LLM sees.
    """
    return Grid(grid).compact()
