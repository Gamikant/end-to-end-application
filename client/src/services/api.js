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



export const trainAndPredict = (formData) => api.post('/train-predict', formData);
