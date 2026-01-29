// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'MCGrad',
  tagline: 'Production-Ready Multicalibration',

  url: 'https://mcgrad.dev',
  baseUrl: '/',
  favicon: 'img/logo_no_text.png',

  organizationName: 'facebookincubator',
  projectName: 'MCGrad',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/facebookincubator/MCGrad/tree/main/website/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
        gtag: {
          trackingID: 'G-SD4H9E0ECC',
          anonymizeIP: true,
        },
      }),
    ],
  ],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      metadata: [
        {name: 'description', content: 'MCGrad is a Python library for production-ready multicalibration. Ensure ML model calibration across all segments, not just globally.'},
        {name: 'keywords', content: 'multicalibration, calibration, machine learning, fairness, MCGrad, Python, LightGBM'},
        {property: 'og:title', content: 'MCGrad - Production-Ready Multicalibration'},
        {property: 'og:description', content: 'Production-ready multicalibration ensuring ML model fairness and accuracy across all segments.'},
        {property: 'og:image', content: 'https://mcgrad.dev/img/logo.png'},
        {property: 'og:type', content: 'website'},
        {name: 'twitter:card', content: 'summary_large_image'},
        {name: 'twitter:title', content: 'MCGrad - Production-Ready Multicalibration'},
        {name: 'twitter:description', content: 'Production-ready multicalibration ensuring ML model fairness and accuracy across all segments.'},
        {name: 'twitter:image', content: 'https://mcgrad.dev/img/logo.png'},
      ],
      navbar: {
        title: 'MCGrad',
        logo: {
          alt: 'MCGrad Logo',
          src: 'img/logo_no_text.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            label: 'Docs',
            position: 'left',
          },
          {
            to: '/docs/tutorials',
            label: 'Tutorials',
            position: 'left',
          },
          {
            href: 'https://mcgrad.readthedocs.io/',
            label: 'API',
            position: 'left',
          },
          {
            href: 'https://github.com/facebookincubator/MCGrad',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        copyright: `Copyright © ${new Date().getFullYear()} Meta Platforms, Inc. · <a href="https://opensource.fb.com/legal/privacy/" target="_blank" rel="noreferrer noopener">Privacy</a> · <a href="https://opensource.fb.com/legal/terms/" target="_blank" rel="noreferrer noopener">Terms</a>`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
