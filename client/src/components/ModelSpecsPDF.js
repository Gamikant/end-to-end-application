import React from 'react';

const PDFViewer = () => {
  return (
    <div style={{ width: '100%', height: '100vh' }}>
      <iframe 
        src="../../public/model.pdf" 
        width="100%"
        height="100%"
        title="PDF Viewer"
        style={{ border: 'none' }}
      >
        <p>Your browser does not support PDF embedding.</p>
      </iframe>
    </div>
  );
};

export default PDFViewer;
