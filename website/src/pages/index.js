/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import CodeBlock from '@theme/CodeBlock';
import Layout from '@theme/Layout';

const features = [
  {
    content: 'Best-in-class calibration quality across a vast number of segments.',
    title: 'State-of-the-art multicalibration',
  },
  {
    content: 'Familiar interface. Pass features, not segments.',
    title: 'Easy to use',
  },
  {
    content: 'Fast to train, low inference overhead, even on web-scale data.',
    title: 'Highly scalable',
  },
  {
    content: 'Likelihood-improving updates with validation-based early stopping.',
    title: 'Safe by design',
  },
  {
    content: 'Designed for real-world deployment and validated at Meta scale.',
    title: 'Proven in production',
  },
];

const Feature = ({title, content}) => {
  return (
    <div className="col feature text--center padding--md">
      <h3>{title}</h3>
      <p>{content}</p>
    </div>
  );
};

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
  <div
    className="get-started-section padding--xl"
    style={{'background-color': 'var(--ifm-color-emphasis-200)'}}
    id="quickstart">
    <h2 className="text--center padding--md">Get Started</h2>
    <ol>
      <li>
        Install MCGrad:
        <CodeBlock
          language="bash"
          showLineNumbers>{`pip install git+https://github.com/facebookincubator/MCGrad.git`}</CodeBlock>
      </li>
      <li>
        Train MCGrad:
        <br />
        <br />
        <CodeBlock language="python" showLineNumbers>
          {codeExample}
        </CodeBlock>
      </li>
    </ol>
  </div>
);

const papertitle = `MCGrad: Multicalibration at Web Scale`;
const paper_bibtex = `
@inproceedings{tax2026mcgrad,
  title = {{MCGrad: Multicalibration at Web Scale}},
  author = {Tax, Niek and Perini, Lorenzo and Linder, Fridolin and Haimovich, Daniel and Karamshuk, Dima and Okati, Nastaran and Vojnovic, Milan and Apostolopoulos, Pavlos Athanasios},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (KDD 2026)},
  year = {2026},
  doi = {10.1145/3770854.3783954}
}`;

const Reference = () => (
  <div
    className="padding--lg"
    id="reference">
    <h2 className='text--center'>Citing MCGrad</h2>
    <p className='text--center'>If you use MCGrad in academic work, please cite:</p>
    <div>
      <a href={`https://arxiv.org/abs/2509.19884`}>{papertitle}</a>
      <CodeBlock className='margin-vert--md'>{paper_bibtex}</CodeBlock>
    </div>
  </div>
);

const MyPage = () => {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <div className="hero text--center" style={{minHeight: '35rem', display: 'flex', alignItems: 'center'}}>
        <div className="container">
          <div className="padding-vert--md">
            <img
              src={useBaseUrl('/img/logo.png')}
              alt="MCGrad Logo"
              className="hero-logo"
            />
            <h1 className="hero__title">MCGrad</h1>
            <p className="hero__subtitle">
              {siteConfig.tagline}
            </p>
          </div>
          <div>
            <Link
              to="/docs/intro"
              className="button button--lg button--primary margin--sm">
              Get Started
            </Link>
            <Link
              to="/docs/why-mcgrad"
              className="button button--lg button--outline button--secondary margin--sm">
              Why MCGrad?
            </Link>
          </div>
        </div>
      </div>
      <div className="padding-top--lg padding-bottom--xl padding-horiz--xl" style={{backgroundColor: 'var(--ifm-background-surface-color)'}}>
        <h2 className="section-title" style={{marginTop: '0', paddingTop: '2rem'}}>Key Features</h2>
        {features && features.length > 0 && (
          <div className="row">
            {features.map(({title, content}) => (
              <Feature
                key={title}
                title={title}
                content={content}
              />
            ))}
          </div>
        )}
      </div>
      <QuickStart />
      <Reference />
    </Layout>
  );
};

export default MyPage;
