# Agent Instructions & Workflow Rules

## PR Review Comment Resolution Workflow

Whenever a GitHub PR review comment is addressed and the fix is committed:
1. **Commit and Push**: Ensure the commit is pushed to the PR branch.
2. **Reply to the Discussion Thread**: Post a reply to the specific comment thread using the GitHub CLI:
   ```bash
   gh api repos/{owner}/{repo}/pulls/{pr_number}/comments/{comment_id}/replies      -f body="Addressed in commit <commit_sha>:
   - <Bullet summary of changes made>"
   ```
3. **Resolve the Review Thread**: Find the GraphQL review thread ID and resolve the discussion thread:
   ```bash
   # Retrieve thread ID
   THREAD_ID=$(gh api graphql -f query='
     query {
       repository(owner: "{owner}", name: "{repo}") {
         pullRequest(number: {pr_number}) {
           reviewThreads(first: 50) {
             nodes {
               id
               isResolved
               comments(first: 1) { nodes { databaseId } }
             }
           }
         }
       }
     }' --jq '.data.repository.pullRequest.reviewThreads.nodes[] | select(.comments.nodes[0].databaseId == <comment_id>) | .id')

   # Resolve the thread
   gh api graphql -f query='
     mutation($id: ID!) {
       resolveReviewThread(input: {threadId: $id}) {
         thread { isResolved }
       }
     }' -f id="$THREAD_ID"
   ```
