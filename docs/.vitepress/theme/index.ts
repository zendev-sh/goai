import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { h } from 'vue'
import { inBrowser } from 'vitepress'
import HomeRaw from './HomeRaw.vue'

import './custom.css'

export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'home-hero-before': () => h(HomeRaw),
    })
  },
  enhanceApp() {
    if (inBrowser) {
      document.documentElement.setAttribute('data-theme-variant', 'dev')
    }
  },
} satisfies Theme
