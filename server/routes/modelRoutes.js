const express = require('express');
const router = express.Router();
const multer = require('multer');
const modelController = require('../controllers/modelController');
const { MlflowClient } = require('mlflow');
const path = require('path');
const fs = require('fs');
const upload = multer({ dest: 'uploads/' });

router.post('/predict', upload.single('file'), modelController.predict);

router.post('/train-predict', upload.fields([
    { name: 'trainData', maxCount: 1 },
    { name: 'testData', maxCount: 1 }
  ]), modelController.trainAndPredict);

router.get('/runs/:runId/artifacts/figures', (req, res) => {
  const runId = req.params.runId;
  const figuresDir = path.join(__dirname, '../artifacts', runId, 'figures');
  fs.readdir(figuresDir, (err, files) => {
    if (err) {
      return res.status(404).json({ error: 'Figures not found for this run.' });
    }
    const imageFiles = files.filter(f =>
      ['.png', '.jpg', '.jpeg'].includes(path.extname(f).toLowerCase())
    );
    res.json(imageFiles.map(f => ({
      path: f,
      url: `/api/runs/${runId}/artifacts/figures/${f}`
    })));
  });
});

router.get('/runs/:runId/artifacts/figures/:filename', (req, res) => {
  const runId = req.params.runId;
  const filename = req.params.filename;
  const filePath = path.join(
    __dirname, 
    '../artifacts', 
    runId, 
    'figures', 
    filename
  );
  
  // Validate file extension for security
  const validExtensions = ['.png', '.jpg', '.jpeg'];
  if (!validExtensions.includes(path.extname(filename).toLowerCase())) {
    return res.status(400).json({ error: 'Invalid file type' });
  }
  
  res.sendFile(filePath, (err) => {
    if (err) {
      res.status(404).json({ error: 'Figure not found' });
    }
  });
});

router.get('/runs/:runId/artifacts/features-dropped', (req, res) => {
  const runId = req.params.runId;
  const filePath = path.join(
    __dirname, 
    '../artifacts', 
    runId, 
    'feature selection', 
    'features_dropped.txt'
  );
  
  fs.readFile(filePath, 'utf8', (err, data) => {
    if (err) {
      return res.status(404).json({ error: 'Features file not found' });
    }
    res.send(data);
  });
});

router.get('/runs/:runId/artifacts/predictions/:filename', (req, res) => {
  const runId = req.params.runId;
  const filePath = path.join(
    __dirname, 
    '../artifacts', 
    runId, 
    'predictions', 
    'confusion_matrix.png' // Fixed filename
  );
  
  res.sendFile(filePath, (err) => {
    if (err) res.status(404).json({ error: 'Confusion matrix not found' });
  });
});



module.exports = router;
