import React from 'react';

export const Results = ({ data }) => {
  return (
    <div className="results-container">
      <h2>Model Results</h2>
      <div className="metrics">
        <h3>Metrics</h3>
        <pre>{JSON.stringify(data.metrics, null, 2)}</pre>
      </div>
      <div className="parameters">
        <h3>Parameters</h3>
        <pre>{JSON.stringify(data.params, null, 2)}</pre>
      </div>
    </div>
  );
};
