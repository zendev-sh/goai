Anthropic's tool search lets a model work with a large tool catalogue without paying for all of it upfront: tools marked `defer_loading` are kept out of the initial request, and the model pulls in the ones it needs through a server-side search tool. For an agent with dozens of tools, that is the difference between spending the context window on definitions and spending it on the task.

The SDK cannot express any of it today:

- The two GA search tools (`tool_search_tool_regex_20251119`, `tool_search_tool_bm25_20251119`) have no factory, so there is no way to enable the search side.
- There is no way to mark a custom tool as deferred — the `defer_loading` wire field is never emitted, so every tool ships in full on every request.
- `tool_search_tool_result` is not in `serverToolResultBlockTypes`, so the result block does not round-trip onto the assistant turn and the model loses what it just looked up.

**Change:** the three pieces, following the patterns already in the file.

- `ToolSearchToolRegex` / `ToolSearchToolBM25` factories, built the same way as the existing provider-defined tools.
- A `DeferLoading bool` on `Tool` / `ToolDefinition`, mapped to `defer_loading` in `convertToolToAPI`. It sits next to the existing tool fields rather than in provider options, since deferred loading is a property of the tool, not of one request.
- `tool_search_tool_result` added to the server-tool result types, so it round-trips like `web_search_tool_result` already does.

Additive and backward compatible: a zero-value `DeferLoading` emits exactly the previous payload, with no `defer_loading` key, and callers that never construct a search tool are unaffected. Both tools are GA, so no beta header is involved.

**Tests:** the factories' wire shape, `defer_loading` present when set and absent when not, and the result block round-tripping onto the assistant turn. Package coverage 98.0%, root 97.6%.
