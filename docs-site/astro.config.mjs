// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import rehypeRewrite from 'rehype-rewrite';

// Deployment configuration
// For GitHub Pages: DEPLOY_TARGET=github-pages
// For Vercel/Netlify: no env vars needed
const isGitHubPages = process.env.DEPLOY_TARGET === 'github-pages';
const site = isGitHubPages
  ? 'https://branes-ai.github.io'
  : (process.env.SITE_URL || 'http://localhost:4321');
const base = isGitHubPages ? '/embodied-ai-architect/' : '/';

// Rehype plugin to rewrite internal links with base path
const rehypeBaseLinks = isGitHubPages ? [
  rehypeRewrite,
  {
    rewrite: (node) => {
      if (node.type === 'element' && node.tagName === 'a' && node.properties?.href) {
        const href = node.properties.href;
        // Rewrite absolute internal links (starting with /) to include base
        if (typeof href === 'string' && href.startsWith('/') && !href.startsWith('/embodied-ai-architect')) {
          node.properties.href = '/embodied-ai-architect' + href;
        }
      }
    }
  }
] : [];

// https://astro.build/config
export default defineConfig({
	server: { host: '0.0.0.0' },
	site: site,
	base: base,
	trailingSlash: 'always',
	markdown: {
		rehypePlugins: rehypeBaseLinks.length ? [rehypeBaseLinks] : [],
	},
	integrations: [
		starlight({
			title: 'Embodied AI Architect',
			description: 'Design, analyze, optimize, and deploy embodied AI solutions',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/branes-ai/embodied-ai-architect' },
			],
			logo: {
				light: './src/assets/logo-light.svg',
				dark: './src/assets/logo-dark.svg',
				replacesTitle: false,
			},
			editLink: {
				baseUrl: 'https://github.com/branes-ai/embodied-ai-architect/edit/main/docs-site/',
			},
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						{ label: 'Introduction', slug: 'getting-started/introduction' },
						{ label: 'Installation', slug: 'getting-started/installation' },
						{ label: 'Quickstart', slug: 'getting-started/quickstart' },
					],
				},
				{
					label: 'Features',
					items: [
						{ label: 'Model Analysis', slug: 'features/model-analysis' },
						{ label: 'Hardware Selection', slug: 'features/hardware-selection' },
						{ label: 'Roofline Analysis', slug: 'features/roofline-analysis' },
						{ label: 'Constraint Checking', slug: 'features/constraint-checking' },
						{ label: 'Codebase Analysis', slug: 'features/codebase-analysis' },
						{ label: 'Deployment', slug: 'features/deployment' },
					],
				},
				{
					label: 'Tutorials',
					autogenerate: { directory: 'tutorials' },
				},
				{
					label: 'Catalog',
					items: [
						{ label: 'Hardware', slug: 'catalog/hardware' },
						{ label: 'Models', slug: 'catalog/models' },
						{ label: 'Sensors', slug: 'catalog/sensors' },
						{ label: 'Operators', slug: 'catalog/operators' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'CLI Commands', slug: 'reference/cli' },
						{ label: 'MCP Tools', slug: 'reference/mcp-tools' },
						{ label: 'Python API', slug: 'reference/api' },
						{ label: 'Constraints', slug: 'reference/constraints' },
					],
				},
				{
					label: 'Troubleshooting',
					autogenerate: { directory: 'troubleshooting' },
				},
			],
			customCss: ['./src/styles/custom.css'],
		}),
	],
});
