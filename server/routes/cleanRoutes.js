// routes/cleanRoutes.js
const express = require('express');
const multer = require('multer');
const router = express.Router();
const fs = require('fs');
const path = require('path');
const modelController = require('../controllers/modelController');

const upload = multer({ dest: 'uploads/' });

router.post('/', upload.single('file'), modelController.predict);

module.exports = router;