<script setup lang="ts">
import {
  Plug,
  Zap,
  Braces,
  Wrench,
  KeyRound,
  Rocket,
} from 'lucide-vue-next'

const features = [
  {
    icon: Plug,
    title: '20 Providers',
    link: '/concepts/providers-and-models',
    details: 'OpenAI, Anthropic, Google, Bedrock, Azure, Vertex, Mistral, xAI, Groq, Cohere, DeepSeek, Ollama, and more. Switch with one line.',
  },
  {
    icon: Zap,
    title: 'Streaming-First',
    link: '/concepts/streaming',
    details: 'Real-time text streaming via Go channels. <code>TextStream()</code> for simple consumption, <code>Stream()</code> for full chunk-level control.',
  },
  {
    icon: Braces,
    title: 'Structured Output',
    link: '/getting-started/structured-output',
    details: '<code>GenerateObject[T]</code> and <code>StreamObject[T]</code> with automatic JSON Schema from Go structs via reflection. No codegen.',
  },
  {
    icon: Wrench,
    title: 'Tools & Agent Loops',
    link: '/concepts/tools',
    details: 'Define tools as Go functions. Auto tool loop with <code>WithMaxSteps()</code>. 20 provider-defined tools (web search, code execution, computer use).',
  },
  {
    icon: KeyRound,
    title: 'TokenSource',
    link: '/concepts/token-source',
    details: 'Dynamic auth for OAuth, cloud IAM, and rotating keys. <code>CachedTokenSource</code> with TTL and mutex-free network refresh. Built for production.',
  },
  {
    icon: Rocket,
    title: 'Faster Than Vercel',
    link: '/getting-started/installation',
    details: '1.4x faster generation, 24x faster cold start, 3.1x less memory. Single binary, zero npm, stdlib only.',
  },
]
</script>

<template>
  <div class="features-section">
    <div class="features-container">
      <div class="features-grid">
        <a v-for="f in features" :key="f.title" :href="f.link" class="feature-card">
          <div class="feature-icon">
            <component :is="f.icon" :size="24" :stroke-width="1.75" />
          </div>
          <h3 class="feature-title" v-html="f.title"></h3>
          <p class="feature-details" v-html="f.details"></p>
        </a>
      </div>
    </div>
  </div>
</template>

<style scoped>
.features-section {
  padding: 2rem 0 0;
}

.features-container {
  max-width: 1152px;
  margin: 0 auto;
  padding: 0 24px;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}

@media (max-width: 960px) {
  .features-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 480px) {
  .features-grid {
    grid-template-columns: 1fr;
  }
}

.feature-card {
  display: block;
  text-decoration: none;
  color: inherit;
  border-radius: 12px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-elv);
  padding: 24px;
  position: relative;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.feature-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--vp-c-brand-1);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.feature-card:hover {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 4px 24px rgba(57, 186, 230, 0.1);
  transform: translateY(-2px);
}

.feature-card:hover::before {
  opacity: 1;
}

.feature-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  border-radius: 10px;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  margin-bottom: 16px;
}

.feature-title {
  font-family: 'Outfit', var(--vp-font-family-base);
  font-weight: 700;
  font-size: 1rem;
  letter-spacing: -0.01em;
  margin: 0 0 8px;
  color: var(--vp-c-text-1);
}

.feature-details {
  font-size: 0.88rem;
  line-height: 1.6;
  color: var(--vp-c-text-2);
  margin: 0;
}

.feature-title :deep(code),
.feature-details :deep(code) {
  font-family: var(--vp-font-family-mono);
  font-size: 0.85em;
  font-weight: 500;
  padding: 2px 6px;
  border-radius: 4px;
  background: var(--vp-code-bg);
  color: var(--vp-code-color);
}
</style>
