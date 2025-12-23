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
    'intro',
    'why-mcgrad',
    'installation',
    'quickstart',
    'methodology',
    'measuring-multicalibration',
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/methods',
        'api/metrics',
        'api/plotting',
        'api/segmentation',
        'api/tuning',
        'api/utils',
      ],
    },
    'contributing',
  ],
};

module.exports = sidebars;
