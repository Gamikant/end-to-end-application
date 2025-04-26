// routes/cleanRoutes.js
const express = require('express');
const multer = require('multer');
const router = express.Router();
const fs = require('fs');
const path = require('path');
const { cleanCSV } = require('../utils/cleaner');
const modelController = require('../controllers/modelController');

const upload = multer({ dest: 'uploads/' });

router.post('/', upload.single('file'), modelController.predict);

// router.post('/', upload.single('testData'), async (req, res) => {
//   try {
//     const cleanedFilePath = await cleanCSV(req.file.path); // implement this
//     res.sendFile(path.resolve(cleanedFilePath)); // send cleaned file back
//   } catch (err) {
//     console.error(err);
//     res.status(500).json({ error: 'Cleaning failed' });
//   }
// });

module.exports = router;