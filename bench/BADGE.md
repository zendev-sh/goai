# Benchmark Badge Snippets

Drop these into the goai README.md when ready (Phase 4.5).

## Shield badges (static, update after bench runs)

```markdown
[![Streaming](https://img.shields.io/badge/streaming-1.1x_faster-brightgreen)](bench/RESULTS.md)
[![Cold Start](https://img.shields.io/badge/cold_start-24x_faster-brightgreen)](bench/RESULTS.md)
[![Memory](https://img.shields.io/badge/memory-3.1x_less-brightgreen)](bench/RESULTS.md)
```

## One-liner for README hero section

```markdown
> **1.1x faster streaming**, **24x faster cold start**, **3.1x less memory** vs Vercel AI SDK
> ([benchmarks](bench/RESULTS.md))
```

## Compact table for README

```markdown
### Performance vs Vercel AI SDK

| Metric | GoAI | Vercel AI SDK | Improvement |
|--------|------|---------------|-------------|
| Streaming throughput | 1.46ms | 1.62ms | 1.1x faster |
| Cold start | 569us | 13.9ms | 24x faster |
| Memory (1 stream) | 220KB | 676KB | 3.1x less |
| GenerateText | 56us | 79us | 1.4x faster |

> Mock HTTP server, identical SSE fixtures, Apple M2. [Full report](bench/RESULTS.md)
```
