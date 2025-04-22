// src/components/ModelDetails.js
import React from 'react';

const ModelDetails = () => {
  return (
    <section id="model-details" className="details-section">
      <h2>Model Specifications</h2>
      
      <div className="model-cards">
        <div className="model-spec classification">
          <h3>Classification Model</h3>
          <ul>
            <li><strong>Type:</strong> CNN</li>
            <li><strong>Accuracy:</strong> 92.4%</li>
            <li><strong>Training Data:</strong> ImageNet</li>
            <li><strong>Use Case:</strong> Image Categorization</li>
          </ul>
        </div>

        <div className="model-spec regression">
          <h3>Regression Model</h3>
          <ul>
            <li><strong>Type:</strong> Gradient Boosting</li>
            <li><strong>R² Score:</strong> 0.89</li>
            <li><strong>Training Data:</strong> Housing Prices</li>
            <li><strong>Use Case:</strong> Price Prediction</li>
          </ul>
        </div>

        <div className="model-spec trainable">
          <h3>Trainable Model</h3>
          <ul>
            <li><strong>Type:</strong> Custom Regression</li>
            <li><strong>Algorithms:</strong> Linear, Ridge, Lasso</li>
            <li><strong>Input Format:</strong> CSV/Excel</li>
            <li><strong>Output:</strong> R², MAE, MSE</li>
          </ul>
        </div>
      </div>
    </section>
  );
};

export default ModelDetails;
