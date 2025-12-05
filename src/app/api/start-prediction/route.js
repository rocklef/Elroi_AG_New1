import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

export async function POST() {
    try {
        const backendDir = path.join(process.cwd(), 'backend');
        const resultFile = path.join(backendDir, 'prediction_result.json');
        const logFile = path.join(backendDir, 'python_log.txt');

        // Clear old files
        if (fs.existsSync(resultFile)) {
            fs.unlinkSync(resultFile);
        }
        if (fs.existsSync(logFile)) {
            fs.unlinkSync(logFile);
        }

        // Write "waiting" status
        fs.writeFileSync(resultFile, JSON.stringify({ status: 'waiting', timestamp: Date.now() }));

        // Spawn Python script with output logging
        const pythonProcess = spawn('python', ['validation2.py'], {
            cwd: backendDir,
            detached: true,
            stdio: ['ignore', 'pipe', 'pipe'] // Use pipe instead of stream
        });

        // Manually pipe output to log file
        const logStream = fs.createWriteStream(logFile, { flags: 'a' });
        pythonProcess.stdout.pipe(logStream);
        pythonProcess.stderr.pipe(logStream);

        pythonProcess.on('error', (err) => {
            fs.appendFileSync(logFile, `\nProcess error: ${err.message}\n`);
        });

        pythonProcess.unref();

        return NextResponse.json({
            started: true,
            message: 'Prediction started. Collecting 20 minutes of data...',
            pid: pythonProcess.pid
        });

    } catch (error) {
        console.error('Error starting prediction:', error);
        return NextResponse.json({
            started: false,
            error: error.message
        }, { status: 500 });
    }
}
