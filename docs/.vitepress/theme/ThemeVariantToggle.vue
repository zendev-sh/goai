<script setup lang="ts">
import { computed } from 'vue'

const variant = (globalThis as any).__goai_variant
const locked = (globalThis as any).__goai_isLocked
const toggle = (globalThis as any).__goai_toggleVariant

const label = computed(() => variant?.value === 'dev' ? 'RAW' : 'CRAFT')
const hint = computed(() => variant?.value === 'dev' ? 'Switch to Craft' : 'Switch to Raw')
const show = computed(() => variant && !locked)
</script>

<template>
  <button
    v-if="show"
    class="theme-variant-toggle"
    :title="hint"
    @click="toggle"
  >
    <span class="toggle-label">{{ label }}</span>
  </button>
</template>

<style scoped>
.theme-variant-toggle {
  display: flex;
  align-items: center;
  margin-left: 8px;
  padding: 3px 10px;
  border-radius: 4px;
  border: 1px solid var(--vp-c-divider);
  background: transparent;
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  cursor: pointer;
  transition: all 0.15s ease;
  white-space: nowrap;
}

.theme-variant-toggle:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}
</style>
