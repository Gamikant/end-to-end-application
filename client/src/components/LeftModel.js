import React, { useState } from 'react';
import { predict } from '../services/api';

const LeftModel = () => {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);

  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append('testData', file);
    
    try {
      const response = await predict(formData);
      setResults(response.data);
    } catch (error) {
      console.error('Prediction error:', error);
    }
  };

  return (
    <div className="model-card">
      <h2>Pre-trained Classification Model</h2>
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button onClick={handleSubmit} disabled={!file}>
        Run Prediction
      </button>
      {results && <pre>{JSON.stringify(results, null, 2)}</pre>}
    </div>
  );
};

export default LeftModel;
