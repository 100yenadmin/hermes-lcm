Closing in favour of the consolidated wave-1 PR #436, which now carries this exact fix
(`embedding_query_spend_max_calls` / `_window_seconds` / `_backoff_seconds`, generous defaults, env-mapped) —
cherry-picked verbatim, with the original tests. Verified present in #436's diff before this closeout
(⟨commit link TBD⟩). Thanks — the fix was correct and is shipping; consolidating so maintainers review one
coherent PR instead of three overlapping ones.
