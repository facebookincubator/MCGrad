import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/MCGrad/__docusaurus/debug',
    component: ComponentCreator('/MCGrad/__docusaurus/debug', '36c'),
    exact: true
  },
  {
    path: '/MCGrad/__docusaurus/debug/config',
    component: ComponentCreator('/MCGrad/__docusaurus/debug/config', '721'),
    exact: true
  },
  {
    path: '/MCGrad/__docusaurus/debug/content',
    component: ComponentCreator('/MCGrad/__docusaurus/debug/content', '971'),
    exact: true
  },
  {
    path: '/MCGrad/__docusaurus/debug/globalData',
    component: ComponentCreator('/MCGrad/__docusaurus/debug/globalData', '092'),
    exact: true
  },
  {
    path: '/MCGrad/__docusaurus/debug/metadata',
    component: ComponentCreator('/MCGrad/__docusaurus/debug/metadata', 'f7d'),
    exact: true
  },
  {
    path: '/MCGrad/__docusaurus/debug/registry',
    component: ComponentCreator('/MCGrad/__docusaurus/debug/registry', '39e'),
    exact: true
  },
  {
    path: '/MCGrad/__docusaurus/debug/routes',
    component: ComponentCreator('/MCGrad/__docusaurus/debug/routes', '8df'),
    exact: true
  },
  {
    path: '/MCGrad/docs',
    component: ComponentCreator('/MCGrad/docs', '850'),
    routes: [
      {
        path: '/MCGrad/docs',
        component: ComponentCreator('/MCGrad/docs', '6ae'),
        routes: [
          {
            path: '/MCGrad/docs',
            component: ComponentCreator('/MCGrad/docs', '011'),
            routes: [
              {
                path: '/MCGrad/docs/api/methods',
                component: ComponentCreator('/MCGrad/docs/api/methods', '1f4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/MCGrad/docs/api/metrics',
                component: ComponentCreator('/MCGrad/docs/api/metrics', 'e72'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/MCGrad/docs/api/plotting',
                component: ComponentCreator('/MCGrad/docs/api/plotting', '6f3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/MCGrad/docs/api/segmentation',
                component: ComponentCreator('/MCGrad/docs/api/segmentation', '78b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/MCGrad/docs/api/tuning',
                component: ComponentCreator('/MCGrad/docs/api/tuning', '79b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/MCGrad/docs/api/utils',
                component: ComponentCreator('/MCGrad/docs/api/utils', 'fe9'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/MCGrad/docs/contributing',
                component: ComponentCreator('/MCGrad/docs/contributing', 'dd7'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/MCGrad/docs/installation',
                component: ComponentCreator('/MCGrad/docs/installation', '31a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/MCGrad/docs/intro',
                component: ComponentCreator('/MCGrad/docs/intro', 'cdb'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/MCGrad/docs/methodology',
                component: ComponentCreator('/MCGrad/docs/methodology', '3a1'),
                exact: true
              },
              {
                path: '/MCGrad/docs/quickstart',
                component: ComponentCreator('/MCGrad/docs/quickstart', '4f3'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/MCGrad/docs/why-mcboost',
                component: ComponentCreator('/MCGrad/docs/why-mcboost', '042'),
                exact: true
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/MCGrad/',
    component: ComponentCreator('/MCGrad/', '583'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
