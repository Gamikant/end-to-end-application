// src/components/ModelSpecs.js
import React from 'react';
import { useLocation } from 'react-router-dom';

const ModelSpecs = () => {
  const location = useLocation();
  const hash = location.hash;

  return (
    <div className="specs-container">
      <h1>Model Specifications</h1>

      <div className="specs-grid">
        <section id="classification" className="model-spec">
          <h2>Classification Model</h2>
          <div className="spec-card">
            <h3>Architecture</h3>
            <ul>
              <li><strong>Type:</strong> Convolutional Neural Network</li>
              <li><strong>Layers:</strong> 10 Convolutional + 3 Dense</li>
              <li><strong>Activation:</strong> ReLU + Softmax</li>
            </ul>
          </div>

          <div className="spec-card">
            <h3>Performance</h3>
            <ul>
              <li><strong>Accuracy:</strong> 92.4% (ImageNet subset)</li>
              <li><strong>Inference Time:</strong> 120ms/image (CPU)</li>
              <li><strong>Training Time:</strong> 4 hours (RTX 3080)</li>
            </ul>
          </div>
        </section>

        <section id="regression" className="model-spec">
          <h2>Regression Model</h2>
          <div className="spec-card">
            <h3>Configuration</h3>
            <ul>
              <li><strong>Algorithms:</strong> Linear, Ridge, Lasso</li>
              <li><strong>Features:</strong> Up to 100 columns</li>
              <li><strong>Normalization:</strong> Automatic scaling</li>
            </ul>
          </div>

          <div className="spec-card">
            <h3>Capabilities</h3>
            <ul>
              <li><strong>Outputs:</strong> R², MAE, MSE, Prediction Plot</li>
              <li><strong>Max Rows:</strong> 100,000 records</li>
              <li><strong>Supported Formats:</strong> CSV, Excel, JSON</li>
            </ul>
          </div>
        </section>
      </div>
    </div>
  );
};

export default ModelSpecs;
