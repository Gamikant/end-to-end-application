// src/components/RightModel.js
import React, { useState } from 'react';
import { trainAndPredict } from '../services/api';

const RightModel = () => {
  const [files, setFiles] = useState({ train: null, test: null });
  const [results, setResults] = useState(null);
  const [isTraining, setIsTraining] = useState(false);

  const handleSubmit = async () => {
    setIsTraining(true);
    const formData = new FormData();
    formData.append('trainData', files.train);
    formData.append('testData', files.test);
    
    try {
      const response = await trainAndPredict(formData);
      setResults(response.data);
    } catch (error) {
      console.error('Training error:', error);
      alert('Training failed. Please check your data.');
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="model-card trainable-model">
      <h2>Trainable Regression Model</h2>
      <div className="file-group">
        <div className="file-upload">
          <label>
            <input
              type="file"
              onChange={(e) => setFiles({...files, train: e.target.files[0]})}
              accept=".csv,.xlsx"
            />
            <span>Upload Training Data</span>
          </label>
          {files.train && (
            <div className="file-info">
              <span>{files.train.name}</span>
              <span>{(files.train.size / 1024).toFixed(2)} KB</span>
            </div>
          )}
        </div>

        <div className="file-upload">
          <label>
            <input
              type="file"
              onChange={(e) => setFiles({...files, test: e.target.files[0]})}
              accept=".csv,.xlsx"
            />
            <span>Upload Test Data</span>
          </label>
          {files.test && (
            <div className="file-info">
              <span>{files.test.name}</span>
              <span>{(files.test.size / 1024).toFixed(2)} KB</span>
            </div>
          )}
        </div>
      </div>

      <button 
        onClick={handleSubmit}
        disabled={!files.train || !files.test || isTraining}
      >
        {isTraining ? 'Training...' : 'Train & Predict'}
      </button>

      {results && (
        <div className="results">
          <h3>Training Results</h3>
          <div className="metrics">
            <div className="metric-card">
              <span>R² Score</span>
              <span className="value">{results.r2_score?.toFixed(3)}</span>
            </div>
            <div className="metric-card">
              <span>MAE</span>
              <span className="value">{results.mae?.toFixed(3)}</span>
            </div>
            <div className="metric-card">
              <span>MSE</span>
              <span className="value">{results.mse?.toFixed(3)}</span>
            </div>
          </div>
          <pre>{JSON.stringify(results, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default RightModel;
