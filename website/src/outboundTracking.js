/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Docusaurus client module that tracks outbound link clicks via GA4.
 *
 * Registered in docusaurus.config.mjs as a clientModule, this runs on every
 * page. It listens for clicks on <a> elements pointing to known external
 * destinations and fires a gtag event with a descriptive label.
 */

const OUTBOUND_RULES = [
  {pattern: '01_mcgrad_core', label: 'colab_01_core'},
  {pattern: '02_calibrating_llm', label: 'colab_02_llm'},
  {pattern: 'colab.research.google.com', label: 'colab_notebook'},
  {pattern: 'github.com/facebookincubator/MCGrad', label: 'github_repo'},
  {pattern: 'mcgrad.readthedocs.io', label: 'api_docs'},
  {pattern: 'youtube.com/watch?v=iAR0NmyS68k', label: 'youtube_video_pydata'},
  {pattern: 'youtube.com/watch?v=-U6yoPAoJF4', label: 'youtube_video_kdd'},
  {pattern: 'arxiv.org/abs/2509.19884', label: 'arxiv_mcgrad'},
  {pattern: 'arxiv.org/abs/2511.11413', label: 'arxiv_matchings'},
  {pattern: 'arxiv.org/abs/2506.11251', label: 'arxiv_measuring_mce'},
  {pattern: 'arxiv.org/pdf/2506.11251', label: 'arxiv_measuring_mce'},
  {pattern: 'arxiv.org/abs/2604.21549', label: 'arxiv_prevalence'},
  {pattern: 'arxiv.org/abs/2602.06773', label: 'arxiv_convergence'},
  {pattern: 'arxiv.org', label: 'arxiv_other'},
  {pattern: 'pypi.org/project/mcgrad', label: 'pypi'},
];

function classifyUrl(href) {
  for (const {pattern, label} of OUTBOUND_RULES) {
    if (href.includes(pattern)) return label;
  }
  return null;
}

export function onRouteDidUpdate() {
  // Re-attach listener on each route change (SPA navigation).
  // Using event delegation on document so we catch dynamically rendered links.
}

if (typeof window !== 'undefined') {
  document.addEventListener('click', (e) => {
    const anchor = e.target.closest('a[href]');
    if (!anchor) return;

    const href = anchor.href;
    if (!href || href.startsWith(window.location.origin)) return;

    const label = classifyUrl(href);
    if (!label) return;

    if (window.gtag) {
      window.gtag('event', 'outbound_click', {
        outbound_destination: label,
        link_url: href,
        link_text: anchor.textContent?.trim()?.slice(0, 100),
        transport_type: 'beacon',
      });
    }
  });
}
