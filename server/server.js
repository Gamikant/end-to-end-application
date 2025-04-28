const express = require('express');
const cors = require('cors');
const multer = require('multer');
const mongoose = require('mongoose');
const config = require('./config/db');
const modelRoutes = require('./routes/modelRoutes');
const errorHandler = require('./middleware/errorHandler');
const cleanRoutes = require('./routes/cleanRoutes');

const app = express();

// Middleware order is critical
app.use(cors({
  origin: 'http://localhost:3000',
  methods: ['POST', 'GET', 'PUT'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Database Connection
mongoose.connect(config.mongoURI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
.then(() => console.log('MongoDB Connected'))
.catch(err => console.log(err));


app.use(require('./middleware/logger'));

// Error Handling
app.use(errorHandler);

// Routes
app.use('/api/models', modelRoutes);
app.use('/api/clean', cleanRoutes);
app.use('/api', modelRoutes);
app.use('/mlflow-artifacts', modelRoutes);

// After routes
app.use((err, req, res, next) => {
  console.error('Server Error:', err.stack);
  res.status(500).json({ 
    error: process.env.NODE_ENV === 'development' ? err.message : 'Internal Server Error'
  });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));