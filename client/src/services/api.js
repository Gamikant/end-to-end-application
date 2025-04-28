import axios from 'axios';

const API_BASE = 'http://localhost:5000/api';

export const predict = (formData) => 
  axios.post(`${API_BASE}/models/predict`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });

export const cleanData = (formData) =>
  axios.post(`${API_BASE}/clean`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    responseType: 'blob'
  });

export const trainAndPredict = (formData) => 
  axios.post(`${API_BASE}/train-predict`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  
export const ShowArtifacts = (runId) => 
  axios.get(`/api/runs/${runId}/artifacts`);

export const getRunFigures = (runId) => 
  axios.get(`${API_BASE}/runs/${runId}/artifacts/figures`);

export const getFeaturesDropped = (runId) => 
  axios.get(`${API_BASE}/runs/${runId}/artifacts/features-dropped`);

export const getConfusionMatrix = (runId) => 
  axios.get(`${API_BASE}/runs/${runId}/artifacts/predictions/confusion_matrix.png`);

export const getRunMetrics = (runId) =>
  axios.get(`/api/runs/${runId}/metrics`);