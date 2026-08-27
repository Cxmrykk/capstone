# AI Agent Instructions: Capstone Progress Logger

To ensure we have a paper trail for the final thesis, you must automatically log our interactions in the `history/` directory.

## When to Log

Create a new log file whenever:

1. We complete a coding task or feature.
2. We fix a bug.
3. I paste terminal outputs, training metrics, or error logs back to you.

## File Naming

- Check the `history/` directory.
- Find the highest numbered file (e.g., `004.md`).
- Create the next sequential file (e.g., `005.md`).

## What to Include in the Log

Do not use a rigid, corporate template. Just write a clean, informal Markdown file containing:

1. **Context:** A quick one-liner about what we were trying to achieve.
2. **Changes & Reasons:** A bulleted list summarizing the files you changed, what you did, and crucially, _why_ you did it.
3. **User Logs:** A code block containing the exact terminal output, error tracebacks, or training metrics (Loss, Exact Match scores, VRAM usage) that I pasted to you.
