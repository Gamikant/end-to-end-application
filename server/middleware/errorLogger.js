const ErrorLog = require('../models/ErrorLog');

module.exports = (error, req, res, next) => {
  ErrorLog.create({
    timestamp: new Date(),
    runId: req.runId,
    message: error.message,
    stack: error.stack,
    stage: req.path.split('/')[2] // Extract pipeline stage from URL
  });
  next(error);
};