// server/routes/modelRoutes.js
const express = require('express');
const router = express.Router();
const modelController = require('../controllers/modelController');

// Add routes
router.post('/predict', modelController.predict);
router.post('/train-predict', modelController.trainAndPredict);

// Critical: Export the router
module.exports = router;
