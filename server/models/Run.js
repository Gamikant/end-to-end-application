const mongoose = require('mongoose');

const runSchema = new mongoose.Schema({
  modelType: String,
  timestamp: { type: Date, default: Date.now },
  metrics: Object,
  params: Object,
  sparkJobId: String,
  mlflowRunId: String
});

module.exports = mongoose.model('Run', runSchema);
