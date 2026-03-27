<script setup lang="ts">
import { ref } from 'vue'

const copied = ref(false)
const cmd = 'go get github.com/zendev-sh/goai@latest'

function copy() {
  navigator.clipboard.writeText(cmd)
  copied.value = true
  setTimeout(() => { copied.value = false }, 2000)
}
</script>

<template>
  <div class="install-block">
    <div class="install-cmd" @click="copy">
      <span class="install-dollar">$</span>
      <code>{{ cmd }}</code>
      <span class="copy-hint" :class="{ copied }">{{ copied ? 'Copied!' : 'Click to copy' }}</span>
    </div>
  </div>
</template>

<style scoped>
.install-block {
  text-align: center;
  margin: -0.5rem auto 3rem;
  padding: 0 1.5rem;
}

.install-cmd {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 12px 24px;
  border-radius: 10px;
  background: var(--vp-code-block-bg);
  border: 1px solid var(--vp-c-divider);
  cursor: pointer;
  transition: border-color 0.2s ease;
}

.dark .install-cmd {
  border-color: rgba(255, 255, 255, 0.08);
}

.install-cmd:hover {
  border-color: var(--vp-c-brand-1);
}

.install-dollar {
  font-family: var(--vp-font-family-mono);
  font-size: 1rem;
  font-weight: 500;
  color: var(--vp-c-text-3);
  user-select: none;
}

.install-cmd code {
  font-family: var(--vp-font-family-mono);
  font-size: 0.95rem;
  font-weight: 500;
  color: var(--vp-c-brand-1);
  letter-spacing: -0.01em;
}

.copy-hint {
  font-family: 'Outfit', sans-serif;
  font-size: 0.7rem;
  font-weight: 500;
  color: var(--vp-c-text-3);
  transition: color 0.2s ease;
}

.install-cmd:hover .copy-hint {
  color: var(--vp-c-text-2);
}

.copy-hint.copied {
  color: #16a34a;
}

.dark .copy-hint.copied {
  color: #22c55e;
}
</style>
