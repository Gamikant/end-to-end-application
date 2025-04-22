// src/components/Home.js
import React from 'react';
import LeftModel from './LeftModel';
import RightModel from './RightModel';
import ModelDetails from './ModelDetails';

const Home = () => {
  return (
    <div className="container">
      <div className="header">
        <h1>ML Model Platform</h1>
        <nav>
          <a href="#model-details">Model Details</a>
          <a href="/login">Admin Login</a>
        </nav>
      </div>

      <div className="model-grid">
        <LeftModel />
        <RightModel />
      </div>

      <ModelDetails />
    </div>
  );
};

export default Home;
