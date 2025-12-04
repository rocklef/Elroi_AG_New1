import { NextResponse } from 'next/server';
import * as XLSX from 'xlsx';
import fs from 'fs';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET() {
    try {
        const filePath = 'C:\\Users\\COE_AT Admin\\Desktop\\software\\SRM PROJECT SUPPORT\\NEW_CHANGE_24_10_25\\Report_20251024H10.xls';

        // Check if file exists
        if (!fs.existsSync(filePath)) {
            return NextResponse.json({ error: 'Excel file not found', isLive: false }, { status: 404 });
        }

        // Force fresh read - clear any cache by reading with timestamp
        const stats = fs.statSync(filePath);
        const fileBuffer = fs.readFileSync(filePath);

        // Parse Excel with fresh options
        const workbook = XLSX.read(fileBuffer, {
            type: 'buffer',
            cellDates: true,
            cellNF: false,
            cellText: false
        });

        // Get the first sheet
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];

        // Convert to JSON with headers
        const jsonData = XLSX.utils.sheet_to_json(worksheet);

        if (jsonData.length === 0) {
            return NextResponse.json({ error: 'No data in Excel file', isLive: false }, { status: 400 });
        }

        // Get headers to find time and temp columns
        const headers = Object.keys(jsonData[0]);
        const timeCol = headers.find(h => h.toLowerCase().includes('time') || h.toLowerCase().includes('date'));
        const tempCol = headers.find(h => h.toLowerCase().includes('temp'));

        if (!tempCol) {
            return NextResponse.json({
                error: 'Temperature column not found',
                columns: headers,
                isLive: false
            }, { status: 400 });
        }

        // Get last row (most recent data)
        const lastRow = jsonData[jsonData.length - 1];
        const temperature = parseFloat(lastRow[tempCol]);

        // Get timestamp
        let timestamp = timeCol ? lastRow[timeCol] : new Date().toLocaleTimeString();

        // Handle Excel date object or serial date
        if (timestamp instanceof Date) {
            timestamp = timestamp.toLocaleTimeString('en-US', { hour12: false });
        } else if (typeof timestamp === 'number') {
            const excelEpoch = new Date(1899, 11, 30);
            const days = Math.floor(timestamp);
            const fractionalDay = timestamp - days;
            const milliseconds = Math.round(fractionalDay * 24 * 60 * 60 * 1000);
            const date = new Date(excelEpoch.getTime() + days * 24 * 60 * 60 * 1000 + milliseconds);
            timestamp = date.toLocaleTimeString('en-US', { hour12: false });
        }

        // Get last 10 readings for the table
        const lastReadings = jsonData.slice(-10).reverse().map(row => {
            let ts = timeCol ? row[timeCol] : 'N/A';
            if (ts instanceof Date) {
                // Format with seconds: HH:MM:SS
                ts = ts.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
            } else if (typeof ts === 'number') {
                // Handle Excel serial date
                const excelEpoch = new Date(1899, 11, 30);
                const days = Math.floor(ts);
                const fractionalDay = ts - days;
                const milliseconds = Math.round(fractionalDay * 24 * 60 * 60 * 1000);
                const date = new Date(excelEpoch.getTime() + days * 24 * 60 * 60 * 1000 + milliseconds);
                ts = date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
            }
            return {
                temperature: parseFloat(row[tempCol]),
                timestamp: String(ts)
            };
        });

        return NextResponse.json({
            temperature: isNaN(temperature) ? 0 : temperature,
            timestamp: String(timestamp),
            totalRows: jsonData.length,
            lastUpdated: new Date().toISOString(),
            fileModified: stats.mtime.toISOString(),
            isLive: true,
            readings: lastReadings,
            columns: { time: timeCol, temp: tempCol }
        }, {
            headers: {
                'Cache-Control': 'no-store, no-cache, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        });

    } catch (error) {
        console.error('Error reading Excel file:', error);
        return NextResponse.json({
            error: 'Failed to read Excel file',
            details: error.message,
            isLive: false
        }, { status: 500 });
    }
}
