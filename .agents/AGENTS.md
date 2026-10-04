# Project Rules

- Do not perform Git commits under any circumstances unless explicitly instructed to do so by the user.
- Always run the full pytest test suite (`.venv/bin/pytest -v`) and application verification checks before code is committed.
- Whenever fixes addressing PR review comments are committed and pushed, always post a response directly to the discussion thread with the commit reference and a summary of the resolution, and mark the corresponding GitHub review thread as resolved.
- Whenever asked to review PR comments, use the following template:
  - **Location**:
  - **Finding**:
  - **Technical Assessment**:
  - **Recommendation**:
- Do not manually request reviews from automated review bots on pull requests; let automated webhook triggers or the user handle review requests.
