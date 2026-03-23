# Project 3: The Cognitive Architecture

**Hard problem: All of them at once**
**Priority: Future / Stretch Goal**
**Hardware: Mac, 48GB RAM, Apple Silicon (MLX)**

## Concept

Build a small agent with distinct cognitive subsystems — working memory, episodic memory, semantic memory, metacognition, tool creation, and a world model. The LLM is the CPU; we build the operating system.

## Architecture

```
┌─────────────────────────────────────────┐
│           METACOGNITION                 │
│   "What do I know? What should I        │
│    learn next? Am I stuck?"             │
├──────────┬──────────┬───────────────────┤
│ WORKING  │ EPISODIC │ SEMANTIC          │
│ MEMORY   │ MEMORY   │ MEMORY            │
│ (context)│ (what    │ (knowledge        │
│          │ happened)│  graph)           │
├──────────┴──────────┴───────────────────┤
│           TOOL CREATION                 │
│   "I don't have a tool for this.        │
│    Let me write one and save it."       │
├─────────────────────────────────────────┤
│           WORLD MODEL                   │
│   "If I do X, I predict Y will happen"  │
└─────────────────────────────────────────┘
```

## Why Deferred

This is the most ambitious project but also the hardest to measure. Projects 1 and 2 provide concrete benchmarks and learnings that inform how to build this properly. Revisit after we have results from the self-improving loop and ARC solver.

## Key Experiment

Drop the agent into an open-ended environment (coding challenges, text games, research tasks). Let it run for days. Questions:
- Does it accumulate useful knowledge?
- Does it get better over time?
- Does it surprise you?
