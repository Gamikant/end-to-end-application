const axios = require('axios');
const Run = require('../models/Run');
const { spawn } = require('child_process');
const path = require('path');
const util = require('util');
const unlink = util.promisify(require('fs').unlink);
const fs = require('fs').promises; // Use promise API

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
  if (!req.files?.trainData || !req.files?.testData) {
    return res.status(400).json({ error: 'Missing training or test files' });
  }

  const uploadDir = path.resolve(__dirname, '../uploads');
  const tempFiles = {
    train: path.resolve(req.files.trainData[0].path),
    test: path.resolve(req.files.testData[0].path),
    validation: path.resolve(uploadDir, 'validation.csv'),
    config: path.resolve(__dirname, '../config/default_hyperparameters.json')
  };

  // === MINIMAL DATA CLEANING STEP ===
  const cleanedTrainPath = path.join(uploadDir, 'cleaned_train.csv');
  const cleanedTestPath = path.join(uploadDir, 'cleaned_test.csv');

  function cleanFile(inputPath, outputPath, originalName) {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python3', [
        path.join(__dirname, '../utils/clean_data.py'),
        inputPath,
        outputPath,
        path.extname(originalName).toLowerCase()
      ]);
      let pythonError = '';
      pythonProcess.stderr.on('data', (data) => pythonError += data.toString());
      pythonProcess.on('close', (code) => {
        if (code !== 0) reject(new Error(pythonError));
        else resolve();
      });
    });
  }

  try {
    await fs.mkdir(uploadDir, { recursive: true });

    // Clean both files in parallel
    await Promise.all([
      cleanFile(tempFiles.train, cleanedTrainPath, req.files.trainData[0].originalname),
      cleanFile(tempFiles.test, cleanedTestPath, req.files.testData[0].originalname)
    ]);

    // Split train/validation with Python, using cleanedTrainPath
    await new Promise((resolve, reject) => {
      const splitProcess = spawn('python3', ['-c', `
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
try:
    df = pd.read_csv("${cleanedTrainPath}")
    train, val = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv("${cleanedTrainPath}", index=False)
    val.to_csv("${tempFiles.validation}", index=False)
    sys.exit(0)
except Exception as e:
    print(f"Split Error: {str(e)}", file=sys.stderr)
    sys.exit(1)
      `]);
      let pythonError = '';
      splitProcess.stderr.on('data', data => pythonError += data.toString());
      splitProcess.stdout.on('data', data => console.log('[Split Python]', data.toString()));

      splitProcess.on('close', code => {
        if (code !== 0) {
          console.error('Split Process Error:', pythonError);
          reject(new Error(`Split failed: ${pythonError || 'Unknown error'}`));
        } else {
          resolve();
        }
      });
    });

    // Update config with cleaned file paths
    const config = JSON.parse(await fs.readFile(tempFiles.config, 'utf8'));
    config.train_file = cleanedTrainPath;
    config.validation_file = tempFiles.validation;
    config.test_file = cleanedTestPath;
    await fs.writeFile(tempFiles.config, JSON.stringify(config, null, 2));

    // Run both pipelines in parallel (unchanged)
    const [featureSelectionResult, finalPipelineResult] = await Promise.all([
      new Promise((resolve, reject) => {
        const pythonProcess = spawn('python3', [
          path.join(__dirname, '../scripts/run_pipeline.py'),
          '--config', tempFiles.config
        ]);
        let output = '';
        let error = '';
        pythonProcess.stdout.on('data', data => output += data.toString());
        pythonProcess.stderr.on('data', data => error += data.toString());
        pythonProcess.on('close', code => {
          if (code !== 0) return reject(error || 'Feature selection failed');
          try {
            const lines = output.trim().split('\n');
            const lastLine = lines.reverse().find(line => line.trim().startsWith('{'));
            const result = JSON.parse(lastLine);
            resolve(result);
          } catch (e) {
            reject('Invalid JSON from feature selection');
          }
        });
      }),
      new Promise((resolve, reject) => {
        const pythonProcess = spawn('python3', [
          path.join(__dirname, '../scripts/run_fp.py'),
          '--config', tempFiles.config
        ]);
        let output = '';
        let error = '';
        pythonProcess.stdout.on('data', data => output += data.toString());
        pythonProcess.stderr.on('data', data => error += data.toString());
        pythonProcess.on('close', code => {
          if (code !== 0) return reject(error || 'Final pipeline failed');
          try {
            const lines = output.trim().split('\n');
            const lastLine = lines.reverse().find(line => line.trim().startsWith('{'));
            const result = JSON.parse(lastLine);
            resolve(result);
          } catch (e) {
            reject(`Invalid JSON from final pipeline: ${output.slice(0, 100)}...`);
          }
        });
      })
    ]);

    // Combine results
    const combinedResult = {
      featureSelection: featureSelectionResult,
      finalPipeline: finalPipelineResult
    };

    res.json(combinedResult);
  } catch (error) {
    await Promise.all([
      fs.unlink(tempFiles.train).catch(() => {}),
      fs.unlink(tempFiles.test).catch(() => {}),
      fs.unlink(tempFiles.validation).catch(() => {}),
      fs.unlink(cleanedTrainPath).catch(() => {}),
      fs.unlink(cleanedTestPath).catch(() => {})
    ]);
    console.error('Training Error:', error);
    const errorMessage = error?.message || String(error);
    res.status(500).json({ error: errorMessage });
  }
};