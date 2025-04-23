// src/components/Home.js
import React from 'react';
import LeftModel from './LeftModel';
import RightModel from './RightModel';
import { useNavigate } from 'react-router-dom';

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="container">
      <div className="header">
        <h1>ML Model Platform</h1>
        {/* <nav>
          <button onClick={() => navigate('/specs')} className="nav-btn">
            Model Specifications
          </button>
          <button onClick={() => navigate('/login')} className="nav-btn">
            Admin Login
          </button>
        </nav> */}
      </div>

      <div className="model-grid">
        <div className="model-section">
          <div className="model-header">
            <h2>Image Classification Model</h2>
            <button 
              onClick={() => navigate('/specs#classification')}
              className="spec-btn"
            >
              View Specifications
            </button>
          </div>
          <p className="model-description">
            Upload images for automatic categorization using our pre-trained CNN model. 
            Supports common image formats and provides confidence scores.
          </p>
          <LeftModel />
        </div>

        <div className="model-section">
          <div className="model-header">
            <h2>Custom Regression Model</h2>
            <button 
              onClick={() => navigate('/specs#regression')}
              className="spec-btn"
            >
              View Specifications
            </button>
          </div>
          <p className="model-description">
            Train and predict using your own datasets. Supports CSV/Excel formats 
            and provides detailed performance metrics (R², MAE, MSE).
          </p>
          <RightModel />
        </div>
      </div>
    </div>
  );
};

export default Home;
