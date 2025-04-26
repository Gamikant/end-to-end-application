import React, { useState } from 'react';
import { cleanData, predict } from '../services/api';

const LeftModel = () => {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleSubmit = async () => {
    if (!file) return;
    setIsProcessing(true);

    try {
      // Step 1: Cleaning - Use 'file' as field name
      const cleanForm = new FormData();
      cleanForm.append('file', file);
      
      const cleanedResponse = await cleanData(cleanForm);
      
      // Step 2: Prediction - Use 'file' as field name
      const predictForm = new FormData();
      const cleanedFile = new File(
        [cleanedResponse.data], 
        'cleaned_data.csv', 
        { type: 'text/csv' }
      );
      predictForm.append('file', cleanedFile);

      const predictionResponse = await predict(predictForm);
      setResults(predictionResponse.data);

    } catch (error) {
      console.error('Error:', error);
      alert(error.response?.data?.error || 'Something went wrong. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="model-card">
      <h2>Pre-trained Classification Model</h2>

      <div className="file-upload">
        <label>
          <input
            type="file"
            onChange={(e) => setFile(e.target.files[0])}
            accept=".csv,.xlsx"
          />
          <span>Upload Test Data</span>
        </label>

        {file && (
          <div className="file-info">
            <span>{file.name}</span>
            <span>{(file.size / 1024).toFixed(2)} KB</span>
          </div>
        )}
      </div>

      <button 
        onClick={handleSubmit} 
        disabled={!file || isProcessing}
        className={isProcessing ? 'processing' : ''}
      >
        {isProcessing ? 'Processing...' : 'Run Prediction'}
      </button>

      {results && (
        <div className="results">
          <h3>Prediction Results</h3>
          <pre>{JSON.stringify(results, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default LeftModel;
