const express = require('express');
const router = express.Router();
const multer = require('multer'); // Import FIRST
const modelController = require('../controllers/modelController');

const upload = multer({ dest: 'uploads/' }); // Initialize AFTER importing

router.post('/predict', upload.single('file'), modelController.predict);
router.post('/train-predict', modelController.trainAndPredict);

module.exports = router;
