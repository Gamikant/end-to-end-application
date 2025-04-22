const axios = require('axios');
const Run = require('../models/Run');

const MLFLOW_TRACKING_URI = 'http://localhost:5000/api/2.0/mlflow';

exports.predict = async (req, res) => {
  try {
    // 1. Create MLflow run
    const runResponse = await axios.post(`${MLFLOW_TRACKING_URI}/runs/create`, {
      experiment_id: "0",
      start_time: Date.now(),
      tags: [{ key: "source", value: "nodejs-backend" }]
    });

    // 2. Log parameters/metrics
    const runId = runResponse.data.run.info.run_id;
    
    await axios.post(`${MLFLOW_TRACKING_URI}/runs/log-parameter`, {
      run_id: runId,
      key: "input_file",
      value: req.file.originalname
    });

    // 3. Store in MongoDB
    const runRecord = new Run({
      run_id: runId,
      model_type: 'classification',
      status: 'RUNNING'
    });
    await runRecord.save();

    res.json({ run_id: runId });
  } catch (error) {
    res.status(500).json({ error: error.message });
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

