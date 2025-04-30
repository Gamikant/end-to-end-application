import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const PreviousRunsHeader = () => {
    const [inputRunId, setInputRunId] = useState('');
    const [fraudRunId, setFraudRunId] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const navigate = useNavigate();
  
    const handleSubmit = async (e) => {
      e.preventDefault();
      if (!inputRunId) return;
  
      setLoading(true);
      setError('');
  
      try {
        const response = await fetch(`http://localhost:5000/api/runs/${inputRunId}/metrics`);
        if (!response.ok) throw new Error('Run not found');
        
        const data = await response.json();
        
        // Extract fraudRunId from finalPipeline.mlflow_run_id
        if (!data?.finalPipeline?.mlflow_run_id) {
          throw new Error('Invalid run data format - missing mlflow_run_id in finalPipeline');
        }
  
        const extractedFraudRunId = data.finalPipeline.mlflow_run_id;
        setFraudRunId(extractedFraudRunId);
        
        // Navigate with inputRunId in path and fraudRunId in query params
        navigate(`/runs/${inputRunId}/features?fraudRunId=${extractedFraudRunId}`);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
  
    return (
      <header className="previous-runs-header">
        <div className="run-id-input-container">
          <form onSubmit={handleSubmit}>
            <input
              type="text"
              value={inputRunId}
              onChange={(e) => setInputRunId(e.target.value)}
              placeholder="Enter Original Run ID"
              className="run-id-input"
            />
            <button type="submit" disabled={loading} className="fetch-button">
              {loading ? 'Loading...' : 'Fetch Run'}
            </button>
          </form>
          {error && <div className="error-message">{error}</div>}
        </div>
  
        {fraudRunId && (
          <nav className="run-tabs">
            <button 
              className="tab-button"
              onClick={() => navigate(`/runs/${inputRunId}/features?fraudRunId=${fraudRunId}`)}
            >
              Feature Selection
            </button>
            <button
              className="tab-button"
              onClick={() => navigate(`/runs/${inputRunId}/results?fraudRunId=${fraudRunId}`)}
            >
              Test Results
            </button>
          </nav>
        )}
      </header>
    );
  };
  
  export default PreviousRunsHeader;