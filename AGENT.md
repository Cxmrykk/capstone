# Development & Experiment Logger Instructions

To maintain an audit trail for experiment reproducibility and development tracking, log major updates in the `history/` directory.

## When to Log

Create a new entry whenever:

1. A module, pipeline stage, or feature implementation is updated.
2. A bug, runtime issue, or environment constraint is resolved.
3. Model evaluation results, training loss curves, or benchmark metrics are collected.

## File Naming

- Inspect the `history/` directory.
- Identify the highest numbered file (e.g., `004.md`).
- Create the next sequential file (e.g., `005.md`).

## Log Structure

Write a structured Markdown record containing:

1. **Context:** A summary of the objective, bug fix, or experiment configuration.
2. **Modifications:** A bulleted list of modified files, code changes, and engineering rationale.
3. **Logs & Metrics:** Terminal output, training telemetry (loss, step count, VRAM usage), or evaluation metrics (GLEU, Exact Match, Execution Accuracy).

When formatting logs within Markdown code blocks, use four backticks: \`\`\`\`markdown.
