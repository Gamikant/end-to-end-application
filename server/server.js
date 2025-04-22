const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const config = require('./config/db');
const modelRoutes = require('./routes/modelRoutes');
const errorHandler = require('./middleware/errorHandler');

const app = express();

// Database Connection
mongoose.connect(config.mongoURI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
.then(() => console.log('MongoDB Connected'))
.catch(err => console.log(err));

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(require('./middleware/logger'));

// Routes
app.use('/api/models', modelRoutes);

// Error Handling
app.use(errorHandler);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
