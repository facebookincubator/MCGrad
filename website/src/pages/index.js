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
    content:
      'MCBoost ensures calibration not just globally, but also across virtually any segment defined using your input features.',
    title: 'Powerful Multi-Calibration',
  },
  {
    content:
      'Borrows information from similar samples like modern ML models, calibrating far more segments than alternatives.',
    title: 'Data Efficient',
  },
  {
    content:
      'Implemented using LightGBM, orders of magnitude faster than heavier NN-based models.',
    title: 'Lightweight & Fast',
  },
  {
    content:
      'A likelihood-improving procedure that can boost model performance with significant PRAUC improvements.',
    title: 'Improved Performance',
  },
];

const Feature = ({title, content}) => {
  return (
    <div className="col col--3 feature text--center padding--md">
      <h3>{title}</h3>
      <p>{content}</p>
    </div>
  );
};

const codeExample = `from multicalibration import MCBoost
import pandas as pd
import numpy as np

# Prepare your data in a DataFrame
df = pd.DataFrame({
    'prediction': np.array([...]),  # Base model predictions
    'label': np.array([...]),       # Ground truth
    'country': [...],                # Categorical features
    'content_type': [...],           # defining segments
    'surface': [...],
})

# Train MCBoost
mcboost = MCBoost()
mcboost.fit(
    df_train=df,
    prediction_column_name='prediction',
    label_column_name='label',
    categorical_feature_column_names=['country', 'content_type', 'surface']
)

# Get multi-calibrated predictions
calibrated_predictions = mcboost.predict(
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
        Train MCBoost:
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
@article{perini2025mcgrad,
  title = {{MCGrad: Multicalibration at Web Scale}},
  author = {Perini, Lorenzo and Haimovich, Daniel and Linder, Fridolin and Tax, Niek and Karamshuk, Dima and Vojnovic, Milan and Okati, Nastaran and Apostolopoulos, Pavlos Athanasios},
  journal = {arXiv preprint arXiv:2509.19884},
  year = {2025},
  note = {To appear in KDD 2026}
}`;

const Reference = () => (
  <div
    className="padding--lg"
    id="reference">
    <h2 className='text--center'>References</h2>
    <div>
      <a href={`https://arxiv.org/pdf/2509.19884`}>{papertitle}</a>
      <CodeBlock className='margin-vert--md'>{paper_bibtex}</CodeBlock>
    </div>
  </div>
);

const SuccessStories = () => (
  <div className="padding--xl" style={{'background-color': 'var(--ifm-color-emphasis-100)'}}>
    <h2 className="section-title">Success Stories</h2>
    <div className="row">
      <div className="col col--4 padding--md">
        <div className="success-story-card">
          <h3>Integrity ReviewBot</h3>
          <p>
            LLM for automated enforcement decisions. MCBoost improved calibration across market and content type,
            reducing multi-calibration error by <strong>6x</strong> and improving PRAUC by <strong>12pp</strong> on average.
            This resulted in <strong>+10pp LLM automation rate</strong>.
          </p>
        </div>
      </div>
      <div className="col col--4 padding--md">
        <div className="success-story-card">
          <h3>Instagram Age Prediction</h3>
          <p>
            Used to gate teen experiences. MCBoost ensured consistent performance across markets and population groups,
            reducing teen miss rate (FNR) by <strong>1.84%</strong> while holding precision constant.
          </p>
        </div>
      </div>
      <div className="col col--4 padding--md">
        <div className="success-story-card">
          <h3>Confidence Aware NSMIEs</h3>
          <p>
            North Star Metric Impact Estimators provide experiment feedback. MCBoost produced confidence-aware
            severity-weighted views prevalence, unblocking label-free precision readouts.
          </p>
        </div>
      </div>
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
              to="/docs/why-mcboost"
              className="button button--lg button--outline button--secondary margin--sm">
              Why MCBoost?
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
      <SuccessStories />
      <QuickStart />
      <Reference />
    </Layout>
  );
};

export default MyPage;
