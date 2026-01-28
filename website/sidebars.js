/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      collapsed: false,
      collapsible: false,
      items: [
        {type: 'doc', id: 'intro', label: 'Why Multicalibration?'},
        'why-mcgrad',
      ],
    },
    {
      type: 'category',
      label: 'How to use',
      collapsed: false,
      collapsible: false,
      items: [
        'installation',
        {type: 'doc', id: 'quickstart', label: 'Getting Started'},
        'tutorials',
        {
          type: 'link',
          label: 'API Reference (ReadTheDocs)',
          href: 'https://mcgrad.readthedocs.io/',
        },
      ],
    },
    {
      type: 'category',
      label: 'Methodology',
      collapsed: false,
      collapsible: false,
      link: {type: 'doc', id: 'methodology'},
      items: [
        {type: 'doc', id: 'methodology', label: 'MCGrad'},
        'measuring-multicalibration',
      ],
    },
    {
      type: 'doc',
      id: 'contributing',
      label: 'Contributing',
    },
  ],
};

module.exports = sidebars;
