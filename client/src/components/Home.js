import React from 'react';
import RightModel from './RightModel';
import { useNavigate } from 'react-router-dom';

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="container">
      <div className="header">
        <h1>ML Model Platform</h1>
      </div>

      <div className="model-grid">
        <div className="model-section">
          <div className="model-header">
            <h2>Custom Fraud Detection</h2>
            <button 
              onClick={() => navigate('/specs#regression')}
              className="spec-btn"
            >
              View Specifications
            </button>
          </div>
          <p className="model-description">
            Train and predict using your own datasets. Supports CSV/Excel formats 
            and provides detailed performance metrics with detailed Reports of Feature Selection and Frauds.
          </p>
          <RightModel />
        </div>
      </div>
    </div>
  );
};

export default Home;
