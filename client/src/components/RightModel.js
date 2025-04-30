// src/components/RightModel.js
import React, { useState } from 'react';
import { trainAndPredict } from '../services/api';
import { useNavigate } from 'react-router-dom';

const RightModel = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState({ train: null, test: null });
  const [results, setResults] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const handleSubmit = async () => {
    setIsTraining(true);
    const formData = new FormData();
    formData.append('trainData', files.train);
    formData.append('testData', files.test);
    
    try {
      const response = await trainAndPredict(formData);
      setResults(response.data);
      // navigate(`/runs/${response.data.featureSelection.mlflow_run_id}/features`);
      navigate(`/runs/${response.data.featureSelection.mlflow_run_id}/features?fraudRunId=${response.data.finalPipeline.mlflow_run_id}`);
    } catch (error) {
      console.error('Training error:', error);
      alert('Training failed. Please check your data.');
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="model-card trainable-model">
      <h2>Trainable Fraud Model</h2>
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
          <h3>Feature Selection Results</h3>
          <div className="dashboard">
            <div className="metrics-panel">
              <div className="metric-item">
                <span className="metric-label">Features Removed</span>
                <span className="metric-value">
                  {results.featureSelection?.features_dropped?.length || 0}
                </span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Features Retained</span>
                <span className="metric-value">
                  {/* If you have this metric */}
                  {results.featureSelection?.metrics?.feature_retention || 'N/A'}
                </span>
              </div>
            </div>

            <div className="mlflow-info">
              <h4>Experiment Tracking</h4>
              <p>
                Feature Selection Run ID:{" "}
                <a
                  href={`http://localhost:5000/#/experiments/0/runs/${results.featureSelection?.mlflow_run_id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {results.featureSelection?.mlflow_run_id}
                </a>
              </p>
              <p>
                Final Pipeline Run ID:{" "}
                <a
                  href={`http://localhost:5000/#/experiments/0/runs/${results.finalPipeline?.mlflow_run_id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {results.finalPipeline?.mlflow_run_id}
                </a>
              </p>
            </div>

            <div className="technical-details">
              <h4>Pipeline Execution Details</h4>
              <button onClick={() => setShowDetails(!showDetails)}>
                {showDetails ? 'Hide' : 'Show'} Technical Logs
              </button>
              {showDetails && (
                <div className="raw-output">
                  <pre>{JSON.stringify(results, null, 2)}</pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}


    </div>
  );
};

export default RightModel;
