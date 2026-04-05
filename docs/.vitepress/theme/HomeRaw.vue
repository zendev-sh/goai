<script setup lang="ts">
import { ref, onMounted } from "vue";

const ready = ref(false);
const installCopied = ref(false);

const providers = [
  { name: "OpenAI", icon: "/providers/openai.svg", link: "/providers/openai" },
  {
    name: "Anthropic",
    icon: "/providers/anthropic.svg",
    link: "/providers/anthropic",
  },
  { name: "Google", icon: "/providers/google.svg", link: "/providers/google" },
  { name: "Bedrock", icon: "/providers/aws.svg", link: "/providers/bedrock" },
  { name: "Azure", icon: "/providers/azure.svg", link: "/providers/azure" },
  { name: "Vertex", icon: "/providers/vertex.svg", link: "/providers/vertex" },
  {
    name: "Mistral",
    icon: "/providers/mistral.svg",
    link: "/providers/mistral",
  },
  { name: "xAI", icon: "/providers/xai.svg", link: "/providers/xai" },
  { name: "Groq", icon: "/providers/groq.svg", link: "/providers/groq" },
  { name: "Cohere", icon: "/providers/cohere.svg", link: "/providers/cohere" },
  {
    name: "DeepSeek",
    icon: "/providers/deepseek.svg",
    link: "/providers/deepseek",
  },
  {
    name: "MiniMax",
    icon: "/providers/minimax.svg",
    link: "/providers/minimax",
  },
  { name: "Ollama", icon: "/providers/ollama.svg", link: "/providers/ollama" },
  {
    name: "Fireworks",
    icon: "/providers/fireworks.svg",
    link: "/providers/fireworks",
  },
  {
    name: "Together",
    icon: "/providers/together.svg",
    link: "/providers/together",
  },
  {
    name: "DeepInfra",
    icon: "/providers/deepinfra.svg",
    link: "/providers/deepinfra",
  },
  {
    name: "OpenRouter",
    icon: "/providers/openrouter.svg",
    link: "/providers/openrouter",
  },
  {
    name: "Perplexity",
    icon: "/providers/perplexity.svg",
    link: "/providers/perplexity",
  },
  {
    name: "Cerebras",
    icon: "/providers/cerebras.svg",
    link: "/providers/cerebras",
  },
  { name: "vLLM", icon: "/providers/vllm.svg", link: "/providers/vllm" },
  {
    name: "Compatible",
    icon: "/providers/compatible.svg",
    link: "/providers/compat",
  },
];

function copyInstall() {
  navigator.clipboard.writeText("go get github.com/zendev-sh/goai@latest");
  installCopied.value = true;
  setTimeout(() => {
    installCopied.value = false;
  }, 2000);
}

onMounted(() => {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      ready.value = true;
    });
  });
});
</script>

<template>
  <div class="rh" :class="{ ready }">
    <!-- ▸ HERO -->
    <section class="rh-hero">
      <div class="rh-hero-inner">
        <div class="rh-hero-left">
          <p class="rh-eyebrow anim a1">Open-source Go SDK</p>
          <h1 class="rh-title anim a2">GoAI</h1>
          <p class="rh-slogan anim a3">AI SDK, the Go way.</p>
          <p class="rh-tagline anim a4">
            One unified API across 22+ providers. Streaming, structured output,
            MCP support, minimal dependencies.
          </p>

          <div class="rh-install anim a5" @click="copyInstall">
            <span class="rh-prompt">$</span>
            <code>go get github.com/zendev-sh/goai@latest</code>
            <span class="rh-copy-state">{{
              installCopied ? "copied!" : "copy"
            }}</span>
          </div>

          <div class="rh-cta anim a6">
            <a href="/getting-started/installation" class="rh-btn-fill"
              >Get Started</a
            >
            <a href="https://github.com/zendev-sh/goai" class="rh-btn-line"
              >GitHub</a
            >
          </div>
        </div>

        <div class="rh-hero-right anim a4">
          <div class="rh-window">
            <div class="rh-window-bar">
              <span class="rh-dot"></span><span class="rh-dot"></span
              ><span class="rh-dot"></span>
              <span class="rh-window-title">main.go</span>
            </div>
            <pre
              class="rh-code"
            ><code><span class="ln"> 1</span>  <span class="kw">package</span> main
<span class="ln"> 2</span>
<span class="ln"> 3</span>  <span class="kw">import</span> (
<span class="ln"> 4</span>      <span class="st">"context"</span>
<span class="ln"> 5</span>      <span class="st">"fmt"</span>
<span class="ln"> 6</span>
<span class="ln"> 7</span>      <span class="st">"github.com/zendev-sh/goai"</span>
<span class="ln"> 8</span>      <span class="st">"github.com/zendev-sh/goai/provider/openai"</span>
<span class="ln"> 9</span>  )
<span class="ln">10</span>
<span class="ln">11</span>  <span class="kw">func</span> <span class="fn">main</span>() {
<span class="ln">12</span>      result, _ := goai.<span class="fn">GenerateText</span>(
<span class="ln">13</span>          context.<span class="fn">Background</span>(),
<span class="ln">14</span>          openai.<span class="fn">Chat</span>(<span class="st">"gpt-4o"</span>),
<span class="ln">15</span>          goai.<span class="fn">WithPrompt</span>(<span class="st">"Explain goroutines."</span>),
<span class="ln">16</span>      )
<span class="ln">17</span>      fmt.<span class="fn">Println</span>(result.Text)
<span class="ln">18</span>  }</code></pre>
          </div>
        </div>
      </div>
    </section>

    <!-- ▸ PROVIDERS -->
    <section class="rh-providers anim a7">
      <div class="rh-providers-inner">
        <p class="rh-section-label">Works with</p>
        <div class="rh-provider-grid">
          <a
            v-for="p in providers"
            :key="p.name"
            :href="p.link"
            class="rh-ptag"
          >
            <img :src="p.icon" :alt="p.name" class="rh-ptag-icon" />
            {{ p.name }}
          </a>
        </div>
      </div>
    </section>

    <!-- ▸ WHAT'S NEW -->
    <section class="rh-whatsnew anim a8">
      <div class="rh-whatsnew-inner">
        <p class="rh-section-label">What's new</p>
        <div class="rh-release-list">
          <a
            href="https://github.com/zendev-sh/goai/releases"
            class="rh-release"
          >
            <span class="rh-release-tag">v0.5.8</span>
            <span class="rh-release-text"
              ><strong>RunPod provider</strong> — Serverless vLLM support.
              Bedrock embeddings. Docs accuracy improvements.</span
            >
          </a>
          <a href="/providers/minimax" class="rh-release">
            <span class="rh-release-tag">v0.5.1</span>
            <span class="rh-release-text"
              ><strong>MiniMax provider</strong> — M2.7, M2.5, M2.1, M2 via
              Anthropic-compatible API. Thinking/reasoning, tool calling,
              streaming.</span
            >
          </a>
          <a href="/concepts/mcp" class="rh-release">
            <span class="rh-release-tag">v0.5.0</span>
            <span class="rh-release-text"
              ><strong>MCP Client</strong> — Connect to any MCP server (stdio,
              HTTP, SSE). Auto-convert tools for GenerateText.</span
            >
          </a>
        </div>
      </div>
    </section>

    <!-- ▸ THREE EXAMPLES -->
    <section class="rh-trio anim a9">
      <div class="rh-trio-inner">
        <a href="/api/core-functions" class="rh-trio-card">
          <div class="rh-trio-head">
            <span class="rh-trio-badge">01</span>
            <span class="rh-trio-name">GenerateText</span>
          </div>
          <pre
            class="rh-trio-code"
          ><code>result, _ := goai.<span class="fn">GenerateText</span>(ctx,
    openai.<span class="fn">Chat</span>(<span class="st">"gpt-4o"</span>),
    goai.<span class="fn">WithPrompt</span>(<span class="st">"Hello."</span>),
)
fmt.<span class="fn">Println</span>(result.Text)</code></pre>
        </a>

        <a href="/concepts/streaming" class="rh-trio-card">
          <div class="rh-trio-head">
            <span class="rh-trio-badge">02</span>
            <span class="rh-trio-name">StreamText</span>
          </div>
          <pre
            class="rh-trio-code"
          ><code>stream, _ := goai.<span class="fn">StreamText</span>(ctx,
    anthropic.<span class="fn">Chat</span>(<span class="st">"claude-sonnet-4-6"</span>),
    goai.<span class="fn">WithMessages</span>(msgs...),
)
<span class="kw">for</span> chunk := <span class="kw">range</span> stream.<span class="fn">TextStream</span>() {
    fmt.<span class="fn">Print</span>(chunk)
}</code></pre>
        </a>

        <a href="/getting-started/structured-output" class="rh-trio-card">
          <div class="rh-trio-head">
            <span class="rh-trio-badge">03</span>
            <span class="rh-trio-name">GenerateObject[T]</span>
          </div>
          <pre
            class="rh-trio-code"
          ><code><span class="kw">type</span> <span class="fn">Recipe</span> <span class="kw">struct</span> {
    Name   <span class="kw">string</span>   <span class="st">`json:"name"`</span>
    Steps  []<span class="kw">string</span> <span class="st">`json:"steps"`</span>
}
recipe, _ := goai.<span class="fn">GenerateObject</span>[Recipe](ctx,
    google.<span class="fn">Chat</span>(<span class="st">"gemini-2.5-flash"</span>),
    goai.<span class="fn">WithPrompt</span>(<span class="st">"Pho recipe"</span>),
)</code></pre>
        </a>
      </div>

      <!-- Row 2: Tools + MCP + TokenSource -->
      <div class="rh-trio-inner">
        <a href="/concepts/tools" class="rh-trio-card">
          <div class="rh-trio-head">
            <span class="rh-trio-badge">04</span>
            <span class="rh-trio-name">Tools & Agent Loop</span>
          </div>
          <pre class="rh-trio-code"><code>calc := goai.Tool{
    Name:    <span class="st">"calc"</span>,
    Execute: <span class="fn">evalMath</span>,
}
result, _ := goai.<span class="fn">GenerateText</span>(ctx, model,
    goai.<span class="fn">WithTools</span>(calc),
    goai.<span class="fn">WithMaxSteps</span>(<span class="st">5</span>), <span class="cm">// auto loop</span>
    goai.<span class="fn">WithPrompt</span>(<span class="st">"127 * 43?"</span>),
)</code></pre>
        </a>

        <a href="/concepts/mcp" class="rh-trio-card">
          <div class="rh-trio-head">
            <span class="rh-trio-badge">05</span>
            <span class="rh-trio-name">MCP Client</span>
          </div>
          <pre
            class="rh-trio-code"
          ><code><span class="cm">// Any MCP server → GoAI tools</span>
c := mcp.<span class="fn">NewClient</span>(<span class="st">"app"</span>, <span class="st">"1.0"</span>,
    mcp.<span class="fn">WithTransport</span>(
        mcp.<span class="fn">NewStdioTransport</span>(
            <span class="st">"npx"</span>, serverArgs),
    ),
)
c.<span class="fn">Connect</span>(ctx)
r, _ := c.<span class="fn">ListTools</span>(ctx, <span class="kw">nil</span>)
mcp.<span class="fn">ConvertTools</span>(c, r.Tools)</code></pre>
        </a>

        <a href="/concepts/token-source" class="rh-trio-card">
          <div class="rh-trio-head">
            <span class="rh-trio-badge">06</span>
            <span class="rh-trio-name">TokenSource</span>
          </div>
          <pre
            class="rh-trio-code"
          ><code><span class="cm">// Auto-refresh OAuth tokens</span>
ts := provider.<span class="fn">CachedTokenSource</span>(
    <span class="kw">func</span>(ctx) (*provider.Token, <span class="kw">error</span>) {
        tok, _ := oauth.<span class="fn">Exchange</span>(ctx, code)
        <span class="kw">return</span> &provider.Token{
            Value: tok.AccessToken,
        }, <span class="kw">nil</span>
    },
)
openai.<span class="fn">Chat</span>(<span class="st">"gpt-4o"</span>,
    openai.<span class="fn">WithTokenSource</span>(ts))</code></pre>
        </a>
      </div>
    </section>

    <!-- ▸ NUMBERS -->
    <section class="rh-numbers anim a10">
      <div class="rh-numbers-inner">
        <div class="rh-num"><em>22</em><span>providers</span></div>
        <div class="rh-num"><em>100+</em><span>models tested</span></div>
        <div class="rh-num"><em>24x</em><span>faster cold start</span></div>
        <div class="rh-num"><em>0</em><span>third-party deps</span></div>
      </div>
    </section>
  </div>
</template>

<style scoped>
/* ======= ANIMATION ORCHESTRATION ======= */
.anim {
  opacity: 0;
  transform: translateY(18px);
  transition:
    opacity 0.5s cubic-bezier(0.16, 1, 0.3, 1),
    transform 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}
.ready .anim {
  opacity: 1;
  transform: translateY(0);
}
.a1 {
  transition-delay: 0.05s;
}
.a2 {
  transition-delay: 0.1s;
}
.a3 {
  transition-delay: 0.18s;
}
.a4 {
  transition-delay: 0.24s;
}
.a5 {
  transition-delay: 0.32s;
}
.a6 {
  transition-delay: 0.38s;
}
.a7 {
  transition-delay: 0.48s;
}
.a8 {
  transition-delay: 0.58s;
}
.a9 {
  transition-delay: 0.68s;
}
.a10 {
  transition-delay: 0.78s;
}

/* ======= TOKENS ======= */
.rh {
  --rh-fg: #111;
  --rh-fg2: #555;
  --rh-fg3: #999;
  --rh-bg: #fcfcfc;
  --rh-bg2: #f3f3f3;
  --rh-line: #e2e2e2;
  --rh-code-bg: #f7f7f7;
  /* syntax - warm muted tones in light */
  --rh-kw: #a626a4;
  --rh-fn: #1a6b50;
  --rh-st: #0451a5;
  --rh-ln: #c5c5c5;
}

:root.dark .rh,
.dark .rh {
  --rh-fg: #e0e0e0;
  --rh-fg2: #8a8a8a;
  --rh-fg3: #555;
  --rh-bg: #0a0a0a;
  --rh-bg2: #141414;
  --rh-line: #222;
  --rh-code-bg: #111;
  --rh-kw: #c678dd;
  --rh-fn: #61afef;
  --rh-st: #98c379;
  --rh-ln: #3a3a3a;
}

/* ======= HERO ======= */
.rh-hero {
  max-width: 1120px;
  margin: 0 auto;
  padding: 5rem 2.5rem 3rem;
}

.rh-hero-inner {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 3.5rem;
  align-items: center;
}

.rh-eyebrow {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--rh-fg3);
  margin: 0 0 1rem;
}

.rh-title {
  font-family: "Space Grotesk", system-ui, sans-serif;
  font-weight: 700;
  font-size: 6rem;
  letter-spacing: -0.07em;
  line-height: 0.85;
  margin: 0 0 0.6rem;
  color: var(--rh-fg);
}

.rh-slogan {
  font-family: "Space Grotesk", system-ui, sans-serif;
  font-weight: 500;
  font-size: 1.5rem;
  letter-spacing: -0.03em;
  color: var(--rh-fg);
  margin: 0 0 1rem;
}

.rh-tagline {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.82rem;
  line-height: 1.7;
  color: var(--rh-fg2);
  margin: 0 0 1.75rem;
  max-width: 440px;
}

/* Install */
.rh-install {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 10px 18px;
  background: var(--rh-code-bg);
  border: 1px solid var(--rh-line);
  cursor: pointer;
  margin-bottom: 1.5rem;
  transition: border-color 0.15s;
}

.rh-install:hover {
  border-color: var(--rh-fg3);
}

.rh-prompt {
  font-family: "IBM Plex Mono", monospace;
  color: var(--rh-fg3);
  user-select: none;
}

.rh-install code {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--rh-fg);
}

.rh-copy-state {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.62rem;
  color: var(--rh-fg3);
}

/* CTA */
.rh-cta {
  display: flex;
  gap: 10px;
}

.rh-btn-fill,
.rh-btn-line {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.78rem;
  font-weight: 600;
  padding: 10px 22px;
  text-decoration: none;
  transition: all 0.12s ease;
}

.rh-btn-fill {
  background: var(--rh-fg);
  color: var(--rh-bg);
}
.rh-btn-fill:hover {
  opacity: 0.8;
}

.rh-btn-line {
  border: 1px solid var(--rh-line);
  color: var(--rh-fg);
  background: transparent;
}
.rh-btn-line:hover {
  border-color: var(--rh-fg3);
}

/* ======= CODE WINDOW ======= */
.rh-window {
  background: var(--rh-code-bg);
  border: 1px solid var(--rh-line);
  overflow: hidden;
  box-shadow: 24px 24px 0 var(--rh-line);
}

.rh-window-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 14px;
  border-bottom: 1px solid var(--rh-line);
  background: var(--rh-bg2);
}

.rh-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--rh-line);
}

.rh-window-title {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.68rem;
  font-weight: 500;
  color: var(--rh-fg3);
  margin-left: 6px;
}

.rh-code {
  margin: 0;
  padding: 16px 18px;
  overflow-x: auto;
}

.rh-code code {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.8rem;
  line-height: 1.65;
  color: var(--rh-fg);
}

.rh-code .ln {
  color: var(--rh-ln);
  user-select: none;
}
.rh-code .kw {
  color: var(--rh-kw);
}
.rh-code .fn {
  color: var(--rh-fn);
}
.rh-code .st {
  color: var(--rh-st);
}

/* ======= PROVIDERS ======= */
.rh-providers {
  border-top: 1px solid var(--rh-line);
  border-bottom: 1px solid var(--rh-line);
  padding: 2.5rem 2.5rem;
}

.rh-providers-inner {
  max-width: 1120px;
  margin: 0 auto;
}

.rh-section-label {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--rh-fg3);
  margin: 0 0 1rem;
  text-align: center;
}

.rh-provider-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
}

a.rh-ptag {
  text-decoration: none;
  color: inherit;
}

.rh-ptag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.72rem;
  font-weight: 500;
  padding: 6px 12px;
  border: 1px solid var(--rh-line);
  color: var(--rh-fg2);
  transition: all 0.12s;
}

.rh-ptag:hover {
  border-color: var(--rh-fg3);
  color: var(--rh-fg);
}

.rh-ptag-icon {
  width: 14px;
  height: 14px;
  opacity: 0.55;
  transition: opacity 0.12s;
}

.rh-ptag:hover .rh-ptag-icon {
  opacity: 0.85;
}

:root.dark .rh-ptag-icon,
.dark .rh-ptag-icon {
  filter: invert(1);
}

/* ======= WHAT'S NEW ======= */
.rh-whatsnew {
  max-width: 1120px;
  margin: 0 auto;
  padding: 2rem 2.5rem;
}

.rh-whatsnew-inner {
  max-width: 1120px;
  margin: 0 auto;
}

.rh-release-list {
  display: flex;
  flex-direction: column;
  gap: 1px;
  background: var(--rh-line);
  border: 1px solid var(--rh-line);
}

a.rh-release {
  text-decoration: none;
  color: inherit;
}

.rh-release {
  display: flex;
  align-items: baseline;
  gap: 16px;
  padding: 14px 20px;
  background: var(--rh-bg);
  transition: background 0.12s;
}

.rh-release:hover {
  background: var(--rh-bg2);
}

.rh-release-tag {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--rh-fn);
  white-space: nowrap;
  min-width: 48px;
}

.rh-release-text {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.78rem;
  line-height: 1.5;
  color: var(--rh-fg2);
}

.rh-release-text strong {
  color: var(--rh-fg);
  font-weight: 600;
}

/* ======= TRIO EXAMPLES ======= */
.rh-trio {
  max-width: 1120px;
  margin: 0 auto;
  padding: 3rem 2.5rem;
}

.rh-trio-inner {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1px;
  background: var(--rh-line);
  border: 1px solid var(--rh-line);
  overflow: hidden;
}

a.rh-trio-card {
  text-decoration: none;
  color: inherit;
  cursor: pointer;
}

.rh-trio-card {
  background: var(--rh-bg);
  display: flex;
  flex-direction: column;
  transition: background 0.12s;
  min-width: 0; /* prevent grid blowout */
}

.rh-trio-card:hover {
  background: var(--rh-bg2);
}

.rh-trio-card .rh-trio-code {
  flex: 1;
}

.rh-trio-head {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--rh-line);
}

.rh-trio-badge {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.6rem;
  font-weight: 700;
  color: var(--rh-fg3);
}

.rh-trio-name {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--rh-fg);
}

.rh-trio-code {
  margin: 0;
  padding: 14px 16px;
  background: var(--rh-code-bg);
  overflow-x: auto;
}

.rh-trio-code code {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.76rem;
  line-height: 1.55;
  color: var(--rh-fg);
  white-space: pre;
  display: block;
  overflow: hidden;
}

.rh-trio-code .kw {
  color: var(--rh-kw);
}
.rh-trio-code .fn {
  color: var(--rh-fn);
}
.rh-trio-code .st {
  color: var(--rh-st);
}
.rh-trio-code .cm {
  color: var(--rh-ln);
  font-style: italic;
}

/* Second trio row: remove double border between rows */
.rh-trio-inner + .rh-trio-inner {
  border-top: 0;
}

/* ======= NUMBERS ======= */
.rh-numbers {
  border-top: 1px solid var(--rh-line);
  border-bottom: 1px solid var(--rh-line);
}

.rh-numbers-inner {
  max-width: 1120px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--rh-line);
  border-left: 1px solid var(--rh-line);
  border-right: 1px solid var(--rh-line);
}

.rh-num {
  text-align: center;
  padding: 2.5rem 1rem;
  background: var(--rh-bg);
}

.rh-num em {
  display: block;
  font-family: "Space Grotesk", system-ui, sans-serif;
  font-style: normal;
  font-size: 3rem;
  font-weight: 700;
  letter-spacing: -0.05em;
  line-height: 1;
  color: var(--rh-fg);
}

.rh-num span {
  display: block;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.68rem;
  font-weight: 500;
  color: var(--rh-fg3);
  margin-top: 8px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

/* ======= RESPONSIVE ======= */
@media (max-width: 960px) {
  .rh-hero-inner {
    grid-template-columns: 1fr;
  }
  .rh-hero-right {
    order: -1;
  }
  .rh-title {
    font-size: 4rem;
  }
  .rh-window {
    box-shadow: 16px 16px 0 var(--rh-line);
  }
  .rh-trio-inner {
    grid-template-columns: 1fr;
  }
  .rh-numbers-inner {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 640px) {
  .rh-hero {
    padding: 3rem 1.5rem 2rem;
  }
  .rh-title {
    font-size: 3rem;
  }
  .rh-slogan {
    font-size: 1.15rem;
  }
  .rh-window {
    box-shadow: 10px 10px 0 var(--rh-line);
  }
  .rh-trio {
    padding: 2rem 1.5rem;
  }
  .rh-providers {
    padding: 2rem 1.5rem;
  }
  .rh-num em {
    font-size: 2.2rem;
  }
}
</style>
