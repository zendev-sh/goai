`DiscoverAuth` resolves an MCP server's OAuth endpoints through a single path: fetch `{origin}/.well-known/oauth-protected-resource`, then `{origin}/.well-known/oauth-authorization-server`. Both at the bare origin, and no other candidate is tried.

That covers the simplest deployment and misses the rest of the discovery chain the MCP authorization spec builds on:

1. **The `WWW-Authenticate` pointer.** RFC 9728 §5.1 has the server advertise its metadata location in the `401` challenge (`resource_metadata="..."`). It is the authoritative answer — it needs no guessing — and it was not read at all.
2. **Path-aware well-known URIs.** A server mounted at `/mcp/` publishes its Protected Resource Metadata at `/.well-known/oauth-protected-resource/mcp/` (RFC 9728 §3.1), not at the origin. Only the bare form was probed.
3. **OpenID Connect Discovery.** Authorization servers commonly publish at `.well-known/openid-configuration` rather than `.well-known/oauth-authorization-server` (RFC 8414). Only the latter was tried, and only in one of its two forms.

The practical effect: a server whose metadata is not at the bare origin cannot be authenticated at all. GitHub's Copilot MCP server is one such case.

**Change:** follow the chain in specificity order, stopping at the first hit.

- `resourceMetadataPointer` sends an unauthenticated probe and reads `resource_metadata` from the `WWW-Authenticate` challenge.
- `protectedResourceWellKnown` returns the path-aware URI first, then the bare-origin one.
- `authServerMetadataURLs` yields both `oauth-authorization-server` and `openid-configuration`, each in host-insert and path-append form.

The existing bare-origin path is still in the list, so servers that work today keep working through the same URLs; the new candidates are only reached when it comes up empty. The legacy fallback of treating the MCP server's own origin as the authorization server is preserved.

**Tests:** cases for the `WWW-Authenticate` pointer, the path-aware well-known, the `openid-configuration` fallback in both forms, and the precedence between them. Package coverage 95.5%.
