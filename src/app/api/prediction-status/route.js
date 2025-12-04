import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export const dynamic = 'force-dynamic';

export async function GET() {
    try {
        const resultFile = path.join(process.cwd(), 'backend', 'prediction_result.json');

        if (!fs.existsSync(resultFile)) {
            return NextResponse.json({
                status: 'idle',
                message: 'No prediction running'
            });
        }

        const data = JSON.parse(fs.readFileSync(resultFile, 'utf8'));

        return NextResponse.json(data, {
            headers: { 'Cache-Control': 'no-store' }
        });

    } catch (error) {
        return NextResponse.json({
            status: 'error',
            error: error.message
        }, { status: 500 });
    }
}
