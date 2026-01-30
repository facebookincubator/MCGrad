/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import CodeBlock from '@theme/CodeBlock';
import Layout from '@theme/Layout';
import useBaseUrl from '@docusaurus/useBaseUrl';

const features = [
  {
    title: 'State-of-the-Art Multicalibration',
    content: 'Best-in-class calibration quality across a vast number of segments.',
  },
  {
    title: 'Easy to Use',
    content: 'Familiar interface. Pass features, not segments.',
  },
  {
    title: 'Highly Scalable',
    content: 'Fast to train, low inference overhead, even on web-scale data.',
  },
  {
    title: 'Safe by Design',
    content: 'Likelihood-improving updates with validation-based early stopping.',
  },
];

const Feature = ({title, content}) => (
  <div className="col feature-card">
    <div className="feature-card__body">
      <h3>{title}</h3>
      <p>{content}</p>
    </div>
  </div>
);

const codeExample = `from mcgrad import methods
import pandas as pd
import numpy as np

# Prepare your data in a DataFrame
df = pd.DataFrame({
    'prediction': np.array([...]),  # Base model predictions
    'label': np.array([...]),        # Ground truth labels
    'country': [...],                 # Categorical features
    'content_type': [...],            # defining segments
    'surface': [...],
})

# Train MCGrad
mcgrad = methods.MCGrad()
mcgrad.fit(
    df_train=df,
    prediction_column_name='prediction',
    label_column_name='label',
    categorical_feature_column_names=['country', 'content_type', 'surface']
)

# Get multicalibrated predictions
calibrated_predictions = mcgrad.predict(
    df=df,
    prediction_column_name='prediction',
    categorical_feature_column_names=['country', 'content_type', 'surface']
)
`;

const QuickStart = () => (
  <div className="section quickstart" id="quickstart">
    <div className="container">
      <div className="text--center">
        <h2>Quickstart</h2>
      </div>
      <div className="row">
        <div className="col col--8 col--offset-2">
          <div className="quickstart__panel">
            <p className="quickstart__label">Install</p>
            <CodeBlock language="bash">pip install mcgrad</CodeBlock>
            <details className="quickstart__details">
              <summary>View minimal code example</summary>
              <CodeBlock language="python" showLineNumbers>
                {codeExample}
              </CodeBlock>
            </details>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const papertitle = `MCGrad: Multicalibration at Web Scale`;
const paper_bibtex = `
@inproceedings{tax2026mcgrad,
  title = {{MCGrad: Multicalibration at Web Scale}},
  author = {Tax, Niek and Perini, Lorenzo and Linder, Fridolin and Haimovich, Daniel
            and Karamshuk, Dima and Okati, Nastaran and Vojnovic, Milan
            and Apostolopoulos, Pavlos Athanasios},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on
               Knowledge Discovery and Data Mining V.1 (KDD 2026)},
  year = {2026},
  doi = {10.1145/3770854.3783954}
}`;

const Reference = () => (
  <div className="section" id="reference">
    <div className="container">
      <div className="text--center">
        <h2>Citing MCGrad</h2>
      </div>
      <div className="row">
        <div className="col col--8 col--offset-2">
          <a href="https://arxiv.org/abs/2509.19884">{papertitle}</a>
          <CodeBlock className='margin-vert--md'>{paper_bibtex}</CodeBlock>
        </div>
      </div>
    </div>
  </div>
);

const MyPage = () => {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <div className="hero hero--compact">
        <div className="container text--center">
          <div className="hero__logo-wrap">
            <img src={useBaseUrl('img/logo_no_text.png')} alt="MCGrad logo" className="hero__logo" />
          </div>
          <h1 className="hero__title hero__title--wordmark">MCGrad</h1>
          <p className="hero__subtitle">Production-ready multicalibration</p>
          <div className="hero__cta">
            <Link
              to="/docs/installation"
              className="button button--lg button--primary">
              Get Started
            </Link>
            <Link
              to="/docs/intro"
              className="button button--lg button--ghost">
              Why Multicalibration?
            </Link>
            <Link
              to="/docs/why-mcgrad"
              className="button button--lg button--ghost">
              Why MCGrad?
            </Link>
          </div>
        </div>
      </div>

      <div className="section section--features" id="features">
        <div className="container">
          <div className="section__header text--center">
            <h2>Key Features</h2>
          </div>
          <div className="row">
            {features.map(({title, content}) => (
              <Feature key={title} title={title} content={content} />
            ))}
          </div>
        </div>
      </div>

      <QuickStart />

      <Reference />
    </Layout>
  );
};

export default MyPage;
