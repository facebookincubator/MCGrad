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
      'MCBoost ensures calibration not just globally, but across virtually any segment defined using input features.',
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
import numpy as np

# Your model predictions and features
predictions = np.array([...])  # Base model predictions
features = pd.DataFrame({
    'country': [...],
    'content_type': [...],
    'surface': [...],
})
labels = np.array([...])  # Ground truth

# Train MCBoost
mcboost = MCBoost()
mcboost.fit(predictions, features, labels)

# Get multi-calibrated predictions
calibrated_predictions = mcboost.predict(predictions, features)
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

const papertitle = `MCBoost: A Tool for Multi-Calibration`;
const paper_bibtex = `
@article{mcboost2024,
  title = {{MCBoost: A Tool for Multi-Calibration}},
  author = {Meta Central Applied Science Team},
  journal = {arXiv preprint arXiv:2509.19884},
  year = {2024},
  url = {https://arxiv.org/pdf/2509.19884}
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
    <h2 className="text--center padding--md">Success Stories</h2>
    <div className="row">
      <div className="col col--4 padding--md">
        <h3>Integrity ReviewBot</h3>
        <p>
          LLM for automated enforcement decisions. MCBoost improved calibration across market and content type,
          reducing multi-calibration error by <strong>6x</strong> and improving PRAUC by <strong>12pp</strong> on average.
          This resulted in <strong>+10pp LLM automation rate</strong>.
        </p>
      </div>
      <div className="col col--4 padding--md">
        <h3>Instagram Age Prediction</h3>
        <p>
          Used to gate teen experiences. MCBoost ensured consistent performance across markets and population groups,
          reducing teen miss rate (FNR) by <strong>1.84%</strong> while holding precision constant.
        </p>
      </div>
      <div className="col col--4 padding--md">
        <h3>Confidence Aware NSMIEs</h3>
        <p>
          North Star Metric Impact Estimators provide experiment feedback. MCBoost produced confidence-aware
          severity-weighted views prevalence, unblocking label-free precision readouts.
        </p>
      </div>
    </div>
  </div>
);

const MyPage = () => {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <div className="hero text--center" style={{height: '30rem'}}>
        <div className="container">
          <div className="padding-vert--md">
            <h1 className="hero__title">MCGrad</h1>
            <p className="hero__subtitle text--secondary">
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
      <div className="padding--xl">
        <h2 className="text--center padding--md">Key Features</h2>
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
