import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import type { PrismTheme } from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

// =====================================================================
// Prism palette — MUST mirror src/css/theme.css.
// CSS custom properties can't be read at build time, so the same hex
// values are duplicated here. When you change a value below, update its
// twin in theme.css (and vice versa). All values are commented with
// their theme.css token name.
// =====================================================================
const PIE_PALETTE = {
  // Light mode (theme.css :root)
  light: {
    bg: '#fdfbf3',  // --pie-code-block-bg  (--ifm-pre-background) — very light beige
    ink: '#2c2722',  // --pie-ink-900
    inkSoft: '#54473b',  // --pie-ink-800
    inkMuted: '#9b8c75',  // a touch lighter than --pie-ink-600 for italic comments
    operator: '#7a6a55',  // operator/url
    primary: '#c25e3d',  // slightly muted from --pie-accent-600 (keywords) — terracotta
    copper: '#6b7a30',  // olive (strings) — warm green-brown contrast
    gold: '#9a5a1f',  // burnt gold (numbers)
    rust: '#a4502d',  // rust (functions)
    plum: '#7a3e6a',  // --pie-plum (class names, properties)
  },
  // Dark mode (theme.css [data-theme='dark'])
  dark: {
    bg: '#2a241d',  // --pie-demi-850 (--pie-tint dark) (--ifm-pre-background)
    ink: '#ede5d2',  // --pie-demi-100
    inkSoft: '#c8bda6',  // --pie-demi-200
    inkMuted: '#8a7c6a',  // mid-warm — for italic comments
    operator: '#a89b85',  // --pie-demi-300
    primary: '#e08a6f',  // lifted terracotta (keywords)
    copper: '#b8c47a',  // sage-light (strings)
    gold: '#d4a45c',  // --pie-amber-light (numbers)
    rust: '#e0a47a',  // --pie-peach-light (functions)
    plum: '#d8a8c0',  // --pie-rose-light (class names, properties)
  },
} as const;

// Build a Prism theme from a palette half. All structural choices
// (which token types share a color, weight bumps, etc.) live here.
function buildPrismTheme(p: typeof PIE_PALETTE.light): PrismTheme {
  return {
    plain: { color: p.ink, backgroundColor: p.bg },
    styles: [
      { types: ['comment', 'prolog', 'doctype', 'cdata'], style: { color: p.inkMuted, fontStyle: 'italic' } },
      { types: ['punctuation'], style: { color: p.inkSoft } },
      { types: ['namespace'], style: { opacity: 0.7 } },
      {
        types: ['keyword', 'tag', 'selector', 'important', 'atrule', 'rule', 'builtin'],
        style: { color: p.primary, fontWeight: '600' }
      },
      {
        types: ['string', 'char', 'attr-value', 'regex', 'inserted'],
        style: { color: p.copper }
      },
      {
        types: ['number', 'boolean', 'symbol', 'constant', 'deleted', 'attr-name'],
        style: { color: p.gold }
      },
      { types: ['function', 'function-variable'], style: { color: p.rust } },
      {
        types: ['class-name', 'maybe-class-name', 'property'],
        style: { color: p.plum, fontWeight: '600' }
      },
      { types: ['operator', 'entity', 'url'], style: { color: p.operator } },
      { types: ['variable'], style: { color: p.ink } },
      { types: ['parameter'], style: { color: p.inkSoft } },
    ],
  };
}

const piePrismLight: PrismTheme = buildPrismTheme(PIE_PALETTE.light);
const piePrismDark: PrismTheme = buildPrismTheme(PIE_PALETTE.dark);

const config: Config = {
  title: 'Pie',
  tagline: 'Programmable System for LLM Serving',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://pie-project.org',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'pie-project', // Usually your GitHub org/user name.
  projectName: 'pie', // Usually your repo name.

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/pie-project/pie/tree/main/website/',
        },
        gtag: process.env.PIE_ENABLE_GTAG === '1'
          ? { trackingID: 'G-D42BHFT9XK', anonymizeIP: false }
          : undefined,
        blog: false,
        theme: {
          // theme.css holds the color palette (single source of truth);
          // custom.css consumes its tokens for component styling.
          customCss: ['./src/css/theme.css', './src/css/custom.css'],
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    // image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Pie',
      logo: {
        alt: 'Pie Logo',
        src: 'img/logo.svg',
        href: '/',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'overviewSidebar',
          position: 'left',
          label: 'Overview',
        },
        {
          type: 'docSidebar',
          sidebarId: 'guideSidebar',
          position: 'left',
          label: 'Guide',
        },
        {
          type: 'docSidebar',
          sidebarId: 'referenceSidebar',
          position: 'left',
          label: 'Reference',
        },
        {
          to: 'models',
          label: 'Models',
          position: 'left',
        },
        {
          to: 'roadmap',
          label: 'Roadmap',
          position: 'left',

        },
        {
          type: 'dropdown',
          label: 'Community',
          position: 'left',
          items: [
            {
              label: 'Getting Involved',
              to: '/community',
            },
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/pie-project/pie/discussions',
            },
          ],
        },
        {
          href: 'https://github.com/pie-project/pie',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    footer: {
      style: 'light',
      links: [
        {
          title: 'Project',
          items: [
            { label: 'GitHub', href: 'https://github.com/pie-project/pie' },
            { label: 'Roadmap', to: '/roadmap' },
            { label: 'Apache 2.0 license', href: 'https://github.com/pie-project/pie/blob/main/LICENSE' },
          ],
        },
        {
          title: 'Docs',
          items: [
            { label: 'Overview', to: '/docs/overview/what-is-pie' },
            { label: 'Guide', to: '/docs/guide/install' },
            { label: 'Reference', to: '/docs/reference/sdk-rust' },
          ],
        },
        {
          title: 'Community',
          items: [
            { label: 'Discussions', href: 'https://github.com/pie-project/pie/discussions' },
            { label: 'Getting involved', to: '/community' },
          ],
        },
      ],
      copyright: `Pie ♥ Open Source<br/>Started as a research project at Yale`,
    },

    prism: {
      theme: piePrismLight,
      darkTheme: piePrismDark,
      additionalLanguages: ['rust'],
    },
  } satisfies Preset.ThemeConfig,

  plugins: [],

  themes: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        language: ['en'],
        indexDocs: true,
        indexBlog: false,
        indexPages: true,
        docsRouteBasePath: '/docs',
        highlightSearchTermsOnTargetPage: true,
        searchResultLimits: 8,
        searchResultContextMaxLength: 60,
        searchBarShortcut: true,
        searchBarShortcutHint: true,
        searchBarPosition: 'right',
      },
    ],
  ],
};

export default config;
