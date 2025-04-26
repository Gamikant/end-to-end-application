// server/utils/cleaner.js
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const { parse } = require('json2csv');

async function cleanCSV(filePath) {
  return new Promise((resolve, reject) => {
    const cleanedRows = [];

    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (row) => {
        const hasData = Object.values(row).some(val => val !== '');
        if (hasData) cleanedRows.push(row);
      })
      .on('end', () => {
        const cleanedCSV = parse(cleanedRows);
        const cleanedPath = path.join(path.dirname(filePath), `cleaned-${path.basename(filePath)}`);
        fs.writeFileSync(cleanedPath, cleanedCSV);
        resolve(cleanedPath);
      })
      .on('error', (err) => {
        reject(err);
      });
  });
}

module.exports = { cleanCSV };
