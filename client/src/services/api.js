import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5000/api/models',
  headers: {
    'Content-Type': 'multipart/form-data'
  }
});

export const predict = (formData) => api.post('/predict', formData);
export const trainAndPredict = (formData) => api.post('/train-predict', formData);
