import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { h, ref, defineComponent } from 'vue'
import { inBrowser } from 'vitepress'
import InstallBlock from './InstallBlock.vue'
import HomeFeatures from './HomeFeatures.vue'
import HomeContent from './HomeContent.vue'
import HomeRaw from './HomeRaw.vue'
import ThemeVariantToggle from './ThemeVariantToggle.vue'

import './custom.css'
import './custom-dev.css'

// Build-time config: GOAI_THEME=dev forces dev theme + hides toggle.
// Set via: GOAI_THEME=dev npm run build
const FORCED_THEME = (import.meta.env.VITE_GOAI_THEME || '') as string
const isLocked = true
const defaultVariant = 'dev'

// Reactive variant state shared across components
const variant = ref<'craft' | 'dev'>(defaultVariant)

function resolveVariant(): 'craft' | 'dev' {
  // If locked by env, always use that
  if (isLocked) return defaultVariant as 'craft' | 'dev'
  if (!inBrowser) return defaultVariant as 'craft' | 'dev'
  const params = new URLSearchParams(window.location.search)
  const fromURL = params.get('theme')
  if (fromURL === 'dev' || fromURL === 'craft') return fromURL
  const stored = sessionStorage.getItem('goai-theme-variant')
  if (stored === 'dev') return 'dev'
  return defaultVariant as 'craft' | 'dev'
}

function applyVariant(v: 'craft' | 'dev') {
  if (!inBrowser) return
  document.documentElement.setAttribute('data-theme-variant', v)
  if (!isLocked) sessionStorage.setItem('goai-theme-variant', v)
  variant.value = v
}

// Expose for ThemeVariantToggle
;(globalThis as any).__goai_variant = variant
;(globalThis as any).__goai_isLocked = isLocked
;(globalThis as any).__goai_toggleVariant = () => {
  if (isLocked) return
  applyVariant(variant.value === 'craft' ? 'dev' : 'craft')
}

// Home content switcher
const HomeSwitcher = defineComponent({
  setup() {
    return () => variant.value === 'dev' ? h(HomeRaw) : null
  }
})

const CraftHomeHero = defineComponent({
  setup() {
    return () => variant.value === 'craft'
      ? [h(InstallBlock), h(HomeFeatures)]
      : null
  }
})

const CraftHomeFeatures = defineComponent({
  setup() {
    return () => variant.value === 'craft' ? h(HomeContent) : null
  }
})

export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'nav-bar-content-after': () => [
        // Hide toggle when theme is locked via env
        isLocked ? null : h(ThemeVariantToggle),
      ],
      'home-hero-before': () => h(HomeSwitcher),
      'home-hero-after': () => h(CraftHomeHero),
      'home-features-after': () => h(CraftHomeFeatures),
    })
  },
  enhanceApp() {
    if (inBrowser) {
      applyVariant(resolveVariant())
    }
  },
} satisfies Theme
