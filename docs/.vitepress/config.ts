import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'GoAI SDK',
  description: 'Go SDK for AI Applications. One API, 22+ LLM Providers.',
  base: process.env.VITEPRESS_BASE || '/',

  sitemap: {
    hostname: 'https://goai.sh',
  },

  lastUpdated: true,

  markdown: {
    theme: {
      light: 'github-light',
      dark: 'github-dark',
    },
  },

  transformHead(context) {
    const head: any[] = []
    if (context.pageData.relativePath === '404.md') return head
    const canonicalUrl = `https://goai.sh/${context.pageData.relativePath.replace(/\.md$/, '.html').replace(/index\.html$/, '')}`
    head.push(['link', { rel: 'canonical', href: canonicalUrl }])
    head.push(['meta', { property: 'og:url', content: canonicalUrl }])

    const title = context.pageData.frontmatter.title || context.pageData.title || 'GoAI SDK'
    const ogTitle = context.pageData.relativePath === 'index.md' ? 'GoAI SDK — Go SDK for AI' : `${title} | GoAI SDK`
    const description = context.pageData.frontmatter.description || context.pageData.description || 'Go SDK for AI Applications. One API, 22+ LLM Providers.'
    head.push(['meta', { property: 'og:title', content: ogTitle }])
    head.push(['meta', { property: 'og:description', content: description }])
    head.push(['meta', { name: 'description', content: description }])
    head.push(['meta', { name: 'twitter:title', content: ogTitle }])
    head.push(['meta', { name: 'twitter:description', content: description }])
    const ogType = context.pageData.relativePath === 'index.md' ? 'website' : 'article'
    head.push(['meta', { property: 'og:type', content: ogType }])
    head.push(['meta', { property: 'og:site_name', content: 'GoAI SDK' }])
    head.push(['meta', { property: 'og:locale', content: 'en_US' }])
    head.push(['meta', { property: 'og:image', content: 'https://goai.sh/goai.png' }])
    head.push(['meta', { name: 'twitter:card', content: 'summary_large_image' }])
    head.push(['meta', { name: 'twitter:image', content: 'https://goai.sh/goai.png' }])

    if (context.pageData.relativePath === 'index.md') {
      head.push(['script', { type: 'application/ld+json' }, JSON.stringify({
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": "GoAI SDK",
        "alternateName": ["goai", "goai.sh"],
        "description": "Go SDK for building AI applications. One SDK, 22+ providers. Supports OpenAI, Anthropic, Google Gemini, AWS Bedrock, Azure, Groq, Mistral, Ollama and more.",
        "url": "https://goai.sh",
        "downloadUrl": "https://github.com/zendev-sh/goai",
        "applicationCategory": "DeveloperApplication",
        "operatingSystem": "Linux, macOS, Windows",
        "programmingLanguage": "Go",
        "license": "https://opensource.org/licenses/MIT",
        "author": {
          "@type": "Organization",
          "name": "zendev",
          "url": "https://zendev.sh"
        },
        "offers": {
          "@type": "Offer",
          "price": "0",
          "priceCurrency": "USD"
        }
      })])
    }

    return head
  },

  head: [
    ['link', { rel: 'icon', type: 'image/png', href: '/goai-icon.png' }],
    ['script', { async: '', src: 'https://www.googletagmanager.com/gtag/js?id=G-0H3DYJ0C2H' }],
    ['script', {}, "window.dataLayer = window.dataLayer || [];\nfunction gtag(){dataLayer.push(arguments);}\ngtag('js', new Date());\ngtag('config', 'G-0H3DYJ0C2H');"],
  ],

  themeConfig: {
    logo: '/goai-icon.png',
    siteTitle: 'GoAI',

    nav: [
      { text: 'Guide', link: '/getting-started/installation' },
      { text: 'Architecture', link: '/architecture' },
      { text: 'Providers', link: '/providers/' },
      { text: 'API', link: '/api/core-functions' },
      { text: 'Examples', link: '/examples' },
      { text: 'Compare', link: '/compare' },
      {
        text: 'Links',
        items: [
          { text: 'GitHub', link: 'https://github.com/zendev-sh/goai' },
          { text: 'GoDoc', link: 'https://pkg.go.dev/github.com/zendev-sh/goai' },
        ],
      },
    ],

    sidebar: {
      '/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Installation', link: '/getting-started/installation' },
            { text: 'Quick Start', link: '/getting-started/quick-start' },
            { text: 'Structured Output', link: '/getting-started/structured-output' },
          ],
        },
        {
          text: 'Architecture',
          link: '/architecture',
        },
        {
          text: 'Concepts',
          items: [
            { text: 'Providers & Models', link: '/concepts/providers-and-models' },
            { text: 'Streaming', link: '/concepts/streaming' },
            { text: 'Tools', link: '/concepts/tools' },
            { text: 'Provider-Defined Tools', link: '/concepts/provider-tools' },
            { text: 'TokenSource', link: '/concepts/token-source' },
            { text: 'MCP Client', link: '/concepts/mcp' },
            { text: 'Error Handling', link: '/concepts/error-handling' },
            { text: 'Multi-Turn Conversations', link: '/concepts/multi-turn' },
            { text: 'Prompt Caching', link: '/concepts/prompt-caching' },
            { text: 'Observability', link: '/concepts/observability' },
          ],
        },
        {
          text: 'Providers',
          collapsed: false,
          items: [
            { text: 'Overview', link: '/providers/' },
            {
              text: 'Tier 1',
              items: [
                { text: 'OpenAI', link: '/providers/openai' },
                { text: 'Anthropic', link: '/providers/anthropic' },
                { text: 'Google', link: '/providers/google' },
                { text: 'AWS Bedrock', link: '/providers/bedrock' },
                { text: 'Azure', link: '/providers/azure' },
                { text: 'Vertex AI', link: '/providers/vertex' },
              ],
            },
            {
              text: 'Tier 2',
              items: [
                { text: 'Cohere', link: '/providers/cohere' },
                { text: 'Mistral', link: '/providers/mistral' },
                { text: 'xAI (Grok)', link: '/providers/xai' },
                { text: 'Groq', link: '/providers/groq' },
                { text: 'DeepSeek', link: '/providers/deepseek' },
                { text: 'MiniMax', link: '/providers/minimax' },
              ],
            },
            {
              text: 'Tier 3',
              items: [
                { text: 'Fireworks', link: '/providers/fireworks' },
                { text: 'Together', link: '/providers/together' },
                { text: 'DeepInfra', link: '/providers/deepinfra' },
                { text: 'OpenRouter', link: '/providers/openrouter' },
                { text: 'Perplexity', link: '/providers/perplexity' },
                { text: 'Cerebras', link: '/providers/cerebras' },
                { text: 'RunPod', link: '/providers/runpod' },
              ],
            },
            {
              text: 'Local / Custom',
              items: [
                { text: 'Ollama', link: '/providers/ollama' },
                { text: 'vLLM', link: '/providers/vllm' },
                { text: 'Compatible', link: '/providers/compat' },
              ],
            },
          ],
        },
        {
          text: 'API Reference',
          items: [
            { text: 'Core Functions', link: '/api/core-functions' },
            { text: 'Types', link: '/api/types' },
            { text: 'Options', link: '/api/options' },
            { text: 'Errors', link: '/api/errors' },
          ],
        },
        {
          text: 'Compare',
          link: '/compare',
        },
        {
          text: 'Examples',
          link: '/examples',
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/zendev-sh/goai' },
    ],

    search: {
      provider: 'local',
    },

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2026 GoAI',
    },

    editLink: {
      pattern: 'https://github.com/zendev-sh/goai/edit/main/docs/:path',
      text: 'Edit this page on GitHub',
    },
  },
})
