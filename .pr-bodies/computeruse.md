Gemini's computer use tool lets the model drive a UI: it sees a screenshot and emits actions (click, type, scroll) that the client executes, returning a fresh screenshot. The actions arrive as ordinary `functionCall`s and their results go back as `functionResponse`s, so the existing tool loop already handles the round trip — but there is no way to declare the tool in the first place.

`Tools` exposes `GoogleSearch`, `URLContext` and `CodeExecution`; computer use is missing, so a caller has to hand-build the `ToolDefinition` and know the wire key it maps to.

**Change:** add a `Tools.ComputerUse` factory following the same shape as the existing provider-defined tools, with two options:

- `WithEnvironment` selects the surface the model controls. The value is the API enum string and is passed through verbatim, so a new environment does not need an SDK release.
- `WithExcludedFunctions` disables specific predefined actions, which is what you want when the client cannot honour some of them (no navigation in a kiosk, for instance).

Both are optional: `Tools.ComputerUse()` emits the bare tool. The definition reaches the wire through the existing `googleProviderTool` path, which camelCases `google.computer_use` into `computerUse` and nests the options under it, so no new serialization is involved.

Purely additive — it adds a constructor and touches nothing on the request path.

**Tests:** the default definition, both options, and the resulting wire shape. Package coverage 98.6%.
