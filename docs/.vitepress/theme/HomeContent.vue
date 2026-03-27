<script setup lang="ts">
import { ref } from 'vue'

const copied = ref(false)

const codeText = `package main

import (
    "context"
    "fmt"

    "github.com/zendev-sh/goai"
    "github.com/zendev-sh/goai/provider/openai"
)

func main() {
    result, _ := goai.GenerateText(context.Background(),
        openai.Chat("gpt-4o"),
        goai.WithPrompt("Explain goroutines in one sentence."),
    )
    fmt.Println(result.Text)
}`

function copyCode() {
  navigator.clipboard.writeText(codeText)
  copied.value = true
  setTimeout(() => { copied.value = false }, 2000)
}
</script>

<template>
  <div class="home-content">
    <div class="home-example">
      <h2>Start in 5 lines</h2>
      <div class="code-wrapper">
        <div class="code-lang">go</div>
        <button class="copy-btn" :class="{ copied }" @click="copyCode">
          {{ copied ? 'Copied!' : 'Copy' }}
        </button>
        <pre><code class="language-go"><span class="kw">package</span> <span class="fn">main</span>

<span class="kw">import</span> (
    <span class="str">"context"</span>
    <span class="str">"fmt"</span>

    <span class="str">"github.com/zendev-sh/goai"</span>
    <span class="str">"github.com/zendev-sh/goai/provider/openai"</span>
)

<span class="kw">func</span> <span class="fn">main</span>() {
    result, _ := goai.<span class="fn">GenerateText</span>(context.<span class="fn">Background</span>(),
        openai.<span class="fn">Chat</span>(<span class="str">"gpt-4o"</span>),
        goai.<span class="fn">WithPrompt</span>(<span class="str">"Explain goroutines in one sentence."</span>),
    )
    fmt.<span class="fn">Println</span>(result.Text)
}</code></pre>
      </div>
    </div>
  </div>
</template>

<style scoped>
.home-content {
  max-width: 720px;
  margin: 0 auto;
  padding: 0 1.5rem;
}

.home-example {
  margin-top: 4rem;
  margin-bottom: 4rem;
}

.home-example h2 {
  font-family: 'Outfit', sans-serif;
  font-weight: 700;
  font-size: 1.5rem;
  letter-spacing: -0.02em;
  text-align: center;
  margin-bottom: 1.5rem;
  color: var(--vp-c-text-1);
}

.code-wrapper {
  position: relative;
  border-radius: 10px;
  border: 1px solid var(--vp-c-divider);
  box-shadow: 0 1px 8px rgba(0, 0, 0, 0.06);
  overflow: hidden;
}

.dark .code-wrapper {
  border-color: rgba(255, 255, 255, 0.08);
  box-shadow: 0 2px 16px rgba(0, 0, 0, 0.4);
}

.code-lang {
  position: absolute;
  top: 10px;
  left: 16px;
  font-family: 'Outfit', sans-serif;
  font-weight: 600;
  font-size: 0.65rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--vp-c-text-3);
}

.copy-btn {
  position: absolute;
  top: 8px;
  right: 12px;
  padding: 4px 12px;
  border-radius: 6px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-3);
  font-family: 'Outfit', sans-serif;
  font-size: 0.72rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.dark .copy-btn {
  border-color: rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.06);
  color: rgba(201, 209, 217, 0.6);
}

.copy-btn:hover {
  color: var(--vp-c-text-1);
  border-color: var(--vp-c-brand-1);
}

.dark .copy-btn:hover {
  background: rgba(255, 255, 255, 0.12);
  color: rgba(201, 209, 217, 0.9);
}

.copy-btn.copied {
  color: #16a34a;
  border-color: rgba(22, 163, 74, 0.3);
}

.dark .copy-btn.copied {
  color: #22c55e;
  border-color: rgba(34, 197, 94, 0.3);
}

.code-wrapper pre {
  margin: 0;
  padding: 2rem 1.5rem 1.5rem;
  background: var(--vp-code-block-bg);
  overflow-x: auto;
}

.code-wrapper code {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.875rem;
  line-height: 1.7;
  color: #24292f;
}

.dark .code-wrapper code {
  color: #c9d1d9;
}

/* Light mode syntax colors (github-light inspired) */
.code-wrapper .kw { color: #cf222e; }
.code-wrapper .fn { color: #6639ba; }
.code-wrapper .str { color: #0a3069; }

.dark .code-wrapper .kw { color: #f97583; }
.dark .code-wrapper .fn { color: #b392f0; }
.dark .code-wrapper .str { color: #9ecbff; }
</style>
