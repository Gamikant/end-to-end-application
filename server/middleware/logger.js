const fs = require('fs');
const path = require('path');

const logsDir = path.join(__dirname, '../logs');

// Ensure logs directory exists
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir);
}

const logStream = fs.createWriteStream(
  path.join(logsDir, 'requests.log'),
  { flags: 'a' }
);

module.exports = (req, res, next) => {
  const logEntry = `
[${new Date().toISOString()}] ${req.method} ${req.path} - ${req.ip} - ${req.get('User-Agent')}
`;

  logStream.write(logEntry);
  next();
};
    