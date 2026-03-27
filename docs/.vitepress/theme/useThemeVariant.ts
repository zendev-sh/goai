/**
 * Theme variant switcher for GoAI docs.
 *
 * ?theme=dev   → "Dev Raw" (opencode.ai / exe.dev inspired)
 * ?theme=craft → "Developer Craft" (default, cyan/blue)
 * no param     → "Developer Craft"
 *
 * Persists choice in sessionStorage so navigation within the site
 * keeps the selected variant without needing ?theme= on every URL.
 */
import { ref, watchEffect, onMounted } from 'vue'
import { inBrowser } from 'vitepress'

const STORAGE_KEY = 'goai-theme-variant'
const ATTR = 'data-theme-variant'

export type ThemeVariant = 'craft' | 'dev'

const current = ref<ThemeVariant>('craft')

function resolveVariant(): ThemeVariant {
  if (!inBrowser) return 'craft'
  const params = new URLSearchParams(window.location.search)
  const fromURL = params.get('theme')
  if (fromURL === 'dev' || fromURL === 'craft') return fromURL
  const stored = sessionStorage.getItem(STORAGE_KEY)
  if (stored === 'dev') return 'dev'
  return 'craft'
}

function applyVariant(v: ThemeVariant) {
  if (!inBrowser) return
  document.documentElement.setAttribute(ATTR, v)
  sessionStorage.setItem(STORAGE_KEY, v)
  current.value = v
}

export function useThemeVariant() {
  onMounted(() => {
    applyVariant(resolveVariant())
  })

  function toggle() {
    applyVariant(current.value === 'craft' ? 'dev' : 'craft')
  }

  return { current, toggle }
}
