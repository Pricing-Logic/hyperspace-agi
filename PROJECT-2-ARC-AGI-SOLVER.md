# Project 2: ARC-AGI Solver via Program Synthesis

**Hard problem: Abstraction + Compositional Generalization**
**Priority: SELECTED — Top 2 Pick (PRIMARY)**
**Hardware: Mac, 48GB RAM, Apple Silicon (MLX)**

## Concept

Build a program synthesis engine that solves ARC-AGI puzzles by generating Python programs, evolving them, and growing its own library of learned abstractions (DSL). The system discovers and names its own primitives.

ARC-AGI is the gold-standard benchmark for general intelligence:
- Humans: ~85%
- Best AI: ~55%
- Prize: $600K at arcprize.org

## Architecture

```
┌────────────────────────────────────────────┐
│              LLM BRAIN (local, MLX)        │
│  "Look at these grids. What's the rule?    │
│   Generate a Python program to transform   │
│   input → output."                         │
├────────────────────────────────────────────┤
│           PROGRAM SYNTHESIZER              │
│  1. LLM generates candidate programs      │
│  2. Test against ALL training pairs        │
│  3. If works on all → apply to test input  │
│  4. If fails → mutate and retry            │
├────────────────────────────────────────────┤
│          GROWING ABSTRACTION DSL           │
│  Learned primitives library:               │
│  - rotate_grid(g, degrees)                 │
│  - flood_fill(g, color, start)             │
│  - find_objects(g) → list of objects       │
│  - mirror(g, axis)                         │
│  - count_by_color(g) → dict               │
│  - [NEW ONES DISCOVERED OVER TIME]         │
├────────────────────────────────────────────┤
│         EVOLUTIONARY ENGINE               │
│  - Genetic programming on program trees    │
│  - LLM-guided mutation (not random)        │
│  - Crossover between successful programs   │
│  - Fitness = % of training pairs solved    │
├────────────────────────────────────────────┤
│         EPISODIC MEMORY                    │
│  - "Puzzle X looked like puzzle Y"         │
│  - "Rotation primitives work on symmetric  │
│    grids"                                  │
│  - Fine-tune LLM on successful solutions   │
└────────────────────────────────────────────┘
```

## Key Design Decisions

- **Model**: Qwen 2.5 3B or similar (must coexist with other loaded model)
- **Program execution**: Sandboxed Python with timeout (prevent infinite loops)
- **DSL growth**: When a program fragment appears in 3+ successful solutions, extract it as a named primitive
- **Evolution**: Population of 50 candidate programs per puzzle, 20 generations max
- **LLM role**: Initial program generation + intelligent mutation suggestions (not brute force)

## Pipeline Per Puzzle

```
1. PERCEIVE: Encode input/output grids as structured data
2. HYPOTHESIZE: LLM generates 10 candidate programs
3. TEST: Run each against all training pairs
4. EVOLVE: For partial matches, mutate and cross over
5. VERIFY: If a program passes all training pairs, apply to test
6. LEARN: Extract successful patterns into DSL
7. REFLECT: Log what worked and why for future puzzles
```

## Success Metrics

- **Primary**: % of ARC-AGI evaluation puzzles solved
- **DSL growth**: Number of useful primitives discovered
- **Transfer**: Does solving puzzle N make puzzle N+1 easier?
- **Generalization**: Performance on held-out puzzle categories

## Why This Is the Primary Pick

- Concrete, measurable benchmark
- Forces us to confront the REAL hard problem (abstraction)
- Program synthesis = fundamentally different from "prompt harder"
- The DSL growth mechanism is a genuine form of learning
- $600K prize if we crack it
- Can start tonight — dataset is public

## Tech Stack

- MLX for local LLM inference
- Python for program synthesis + execution sandbox
- ARC-AGI dataset (JSON format, public)
- SQLite for experiment logging
- AST module for program tree manipulation
- Genetic programming library or custom evolution engine
