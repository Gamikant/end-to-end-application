const axios = require('axios');
const Run = require('../models/Run');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const util = require('util');

const unlink = util.promisify(fs.unlink);

const MLFLOW_TRACKING_URI = 'http://localhost:5000/api/2.0/mlflow';

exports.predict = async (req, res) => {
  try {
    if (!req.file) throw new Error("No file uploaded");

    const inputPath = req.file.path;
    const outputPath = path.join(__dirname, 'cleaned_data.csv');

    const originalExt = path.extname(req.file.originalname).toLowerCase();
    
    // Spawn the Python process
    const pythonProcess = spawn('python3', [
      path.join(__dirname, '../utils/clean_data.py'),
      inputPath,
      outputPath,
      originalExt
    ]);

    let pythonOutput = '';
    let pythonError = '';

    pythonProcess.stdout.on('data', (data) => {
      pythonOutput += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      pythonError += data.toString();
    });

    // In modelController.js
    pythonProcess.on('close', (code) => {
      // Only treat as error if exit code is non-zero
      if (code !== 0) {
        console.error('Python error:', pythonError);
        return res.status(500).json({ error: 'Python script failed', details: pythonError });
      }
    
      // Success: Process cleaned data
      fs.readFile(outputPath, 'utf8', (err, cleanedData) => {
        if (err) return res.status(500).json({ error: 'Failed to read cleaned data' });
        res.header('Content-Type', 'text/csv').send(cleanedData);
      });
    });
  } catch (error) {
    
    console.error('Prediction Error:', error);
    res.status(500).json({ 
      error: process.env.NODE_ENV === 'development' 
        ? error.message 
        : 'Processing failed' 
    });
  }
};

exports.trainAndPredict = async (req, res) => {
  try {
    // 1. Create MLflow run for training
    const runResponse = await axios.post(`${MLFLOW_TRACKING_URI}/runs/create`, {
      experiment_id: "1",
      start_time: Date.now(),
      tags: [{ key: "source", value: "nodejs-training" }]
    });

    const runId = runResponse.data.run.info.run_id;

    // 2. Log training parameters
    await axios.post(`${MLFLOW_TRACKING_URI}/runs/log-parameter`, {
      run_id: runId,
      key: "training_files",
      value: req.files.map(f => f.originalname).join(',')
    });

    // 3. Store in MongoDB
    const runRecord = new Run({
      run_id: runId,
      model_type: 'trainable-classification',
      status: 'TRAINING'
    });
    await runRecord.save();

    res.json({ run_id: runId });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};

