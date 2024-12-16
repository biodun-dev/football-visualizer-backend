const express = require('express');
const fs = require('fs');
const path = require('path');
const csvParser = require('csv-parser');
const cors = require('cors');

const app = express();
app.use(cors());

// Base directory for CSV files
const BASE_PATH = '/home/biodun/footballanalysis/Bird\'s eye view/inference/output';

app.get('/api/yolo-detections', (req, res) => {
    const filePath = path.join(BASE_PATH, 'yolo_detections.csv');
    const results = [];
    fs.createReadStream(filePath)
        .pipe(csvParser())
        .on('data', (data) => results.push(data))
        .on('end', () => {
            res.json(results);
        })
        .on('error', (err) => {
            res.status(500).json({ error: 'Error reading YOLO detections CSV' });
        });
});

// Serve events
app.get('/api/events', (req, res) => {
    const filePath = path.join(BASE_PATH, 'events.csv');
    const results = [];
    fs.createReadStream(filePath)
        .pipe(csvParser())
        .on('data', (data) => results.push(data))
        .on('end', () => {
            res.json(results);
        })
        .on('error', (err) => {
            res.status(500).json({ error: 'Error reading events CSV' });
        });
});

// Start the server
const PORT = 3001;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
