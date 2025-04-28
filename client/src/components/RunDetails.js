import React, { useEffect, useState } from 'react';
import { useParams, NavLink, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { getRunFigures, getFeaturesDropped, getConfusionMatrix } from '../services/api';

function prettifyFigureName(filename) {
  return filename
    .replace(/\.[^/.]+$/, '')
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

const FeatureSelectionDetails = ({ loading, features, artifacts, runId }) => {
  const [zoomedIdx, setZoomedIdx] = useState(null);
  const API_BASE = 'http://localhost:5000/api';
  return (
    <section className="feature-selection-summary">
      <h3>Feature Selection Summary</h3>
      {loading ? (
        <div>Loading...</div>
      ) : (
        <>
          <div className="features-table">
            <h4>Dropped Features ({features.length})</h4>
            <div className="table-responsive">
              <table className="small-table">
                <thead>
                  <tr>
                    <th>Sl. No.</th>
                    <th>Feature Name</th>
                  </tr>
                </thead>
                <tbody>
                  {features.map((feature, index) => (
                    <tr key={index}>
                      <td>{index + 1}</td>
                      <td>{feature}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <div className="figure-grid">
            {artifacts.length === 0 ? (
              <div>No figures found for this run.</div>
            ) : (
              artifacts.map((artifact, index) => (
                <div key={index} className="figure-card">
                  <img
                    src={`${API_BASE}/runs/${runId}/artifacts/figures/${artifact.path}`}
                    alt={artifact.path}
                    className={`large-figure${zoomedIdx === index ? ' zoomed' : ''}`}
                    onClick={() => setZoomedIdx(zoomedIdx === index ? null : index)}
                    style={{ cursor: 'zoom-in' }}
                  />
                  <div className="figure-meta">{prettifyFigureName(artifact.path)}</div>
                </div>
              ))
            )}
          </div>
        </>
      )}
    </section>
  );
};

// --- Test Results Tab: show confusion matrix image ---
// src/components/RunDetails.js
const TestResultsDetails = () => {
  const location = useLocation();
  const [confusionMatrixUrl, setConfusionMatrixUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const params = new URLSearchParams(location.search);
  const fraudRunId = params.get('fraudRunId');
  const API_BASE = 'http://localhost:5000/api';

  useEffect(() => {
    if (!fraudRunId) {
      setError('No Fraud Detection Run ID found.');
      setLoading(false);
      return;
    }
    setLoading(true);
    fetch(`${API_BASE}/runs/${fraudRunId}/artifacts/predictions/confusion_matrix.png`)
      .then(response => {
        if (!response.ok) throw new Error('Confusion matrix not found');
        return response.blob();
      })
      .then(blob => {
        setConfusionMatrixUrl(URL.createObjectURL(blob));
        setError('');
      })
      .catch(err => {
        setError('Error fetching confusion matrix.');
        setConfusionMatrixUrl(null);
      })
      .finally(() => setLoading(false));
  }, [fraudRunId]);

  return (
    <section className="test-results-summary">
      <h3>Test Results</h3>
      {loading ? (
        <div>Loading confusion matrix...</div>
      ) : error ? (
        <div style={{ color: 'red' }}>{error}</div>
      ) : confusionMatrixUrl ? (
        <div style={{ textAlign: 'center' }}>
          <img
            src={confusionMatrixUrl}
            alt="Confusion Matrix"
            style={{ maxWidth: '600px', margin: '20px 0' }}
          />
        </div>
      ) : (
        <div>No confusion matrix available</div>
      )}
    </section>
  );
};



const RunDetails = () => {
  const { runId } = useParams();
  const [artifacts, setArtifacts] = useState([]);
  const [features, setFeatures] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const location = useLocation();

  useEffect(() => {
    if (!runId) return;
    setLoading(true);
    setError('');
    
    Promise.all([
      getRunFigures(runId),
      getFeaturesDropped(runId),
    ])
      .then(([figuresResponse, featuresResponse]) => {
        setArtifacts(figuresResponse.data);
        setFeatures(featuresResponse.data.split('\n').filter(Boolean));
      })
      .catch((err) => {
        setError('Error fetching run details.');
        console.error(err);
      })
      .finally(() => setLoading(false));
  }, [runId]);

  // Default to /features tab if user lands on /runs/:runId
  if (location.pathname === `/runs/${runId}`) {
    return <Navigate to={`/runs/${runId}/features${location.search}`} replace />;
  }

  return (
    <div className="run-details">
      <h2>Run Details: {runId}</h2>
      <nav className="run-subnav">
        <NavLink
          to={`/runs/${runId}/features${location.search}`}
          className={({ isActive }) => isActive ? 'active' : ''}
        >
          Feature Selection
        </NavLink>
        <NavLink
          to={`/runs/${runId}/results${location.search}`}
          className={({ isActive }) => isActive ? 'active' : ''}
        >
          Test Results
        </NavLink>
      </nav>
      {error && <div className="error">{error}</div>}
      <Routes>
        <Route
          path="features"
          element={
            <FeatureSelectionDetails
              loading={loading}
              features={features}
              artifacts={artifacts}
              runId={runId}
            />
          }
        />
        <Route path="results" element={<TestResultsDetails />} />
        {/* Optionally, redirect unknown subroutes */}
        <Route path="*" element={<Navigate to={`/runs/${runId}/features${location.search}`} replace />} />
      </Routes>
    </div>
  );
};

export default RunDetails;
