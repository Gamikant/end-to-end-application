import React from 'react';

const ModelSpecs = () => (
  <div className="specs-container">
    <h1>Model Specifications</h1>
    <div style={{
      width: '100%',
      maxWidth: 900,
      margin: '2rem auto',
      background: '#fff',
      borderRadius: 8,
      boxShadow: '0 2px 16px #0001',
      padding: '2rem'
    }}>
      <object
        data="/model.pdf"
        type="application/pdf"
        width="100%"
        height="900px"
        style={{ border: '1px solid #eee', borderRadius: 8 }}
      >
        <p>
          This browser does not support PDF viewing.
          <a href="/model.pdf" target="_blank" rel="noopener noreferrer">Download PDF</a>
        </p>
      </object>
    </div>
  </div>
);

export default ModelSpecs;
