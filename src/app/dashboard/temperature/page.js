'use client'

import { useState, useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { supabase } from '@/lib/supabase'
import Sidebar from '@/components/Sidebar'
import TopNav from '@/components/TopNav'

// Temperature Gauge Component
function TemperatureGauge({ value = 41.9, min = 25, max = 60, threshold = 31.7, onThresholdChange }) {
  const [showThresholdInput, setShowThresholdInput] = useState(false)
  const [tempThreshold, setTempThreshold] = useState(threshold)

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v))
  const mapTempToAngle = (t) => {
    const clamped = clamp(t, min, max)
    const ratio = (clamped - min) / (max - min)
    // REVERSED: High temp = left (180°), Low temp = right (0°)
    return 180 - (180 * ratio) // Flip the angle
  }
  const polarToCartesian = (cx, cy, r, angleDeg) => {
    const rad = (Math.PI * angleDeg) / 180
    return { x: cx + r * Math.cos(rad), y: cy - r * Math.sin(rad) }
  }
  const describeArc = (cx, cy, r, startAngle, endAngle) => {
    const start = polarToCartesian(cx, cy, r, startAngle)
    const end = polarToCartesian(cx, cy, r, endAngle)
    const largeArcFlag = endAngle - startAngle <= 180 ? 0 : 1
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`
  }

  const width = 360
  const height = 220
  const cx = width / 2
  const cy = height
  const radius = 160

  // COOLING SYSTEM REVERSED: As temp decreases, needle moves RIGHT
  // Safe zone (dark green): from max to threshold (LEFT side - higher temps)
  const safeStart = mapTempToAngle(max) // Left side (180°)
  const safeEnd = mapTempToAngle(threshold)

  // Danger zone (dark red): from threshold to min (RIGHT side - lower temps)
  const dangerStart = mapTempToAngle(threshold)
  const dangerEnd = mapTempToAngle(min) // Right side (0°)

  const needleAngle = mapTempToAngle(value)
  const needleEnd = polarToCartesian(cx, cy, radius - 16, needleAngle)
  const thEnd = polarToCartesian(cx, cy, radius - 8, mapTempToAngle(threshold))
  const exceeded = value <= threshold // Alert when temperature goes BELOW threshold

  const handleApplyThreshold = () => {
    const newValue = parseFloat(tempThreshold)
    if (!isNaN(newValue)) {
      onThresholdChange?.(newValue)
      setShowThresholdInput(false)
    } else {
      alert('Please enter a valid number')
    }
  }

  useEffect(() => {
    setTempThreshold(threshold)
  }, [threshold])

  return (
    <div className={`rounded-2xl p-6 backdrop-blur-md shadow-xl border ${
      exceeded ? 'border-red-600 ring-8 ring-red-500/60 shadow-[0_0_40px_rgba(239,68,68,0.6)] bg-gradient-to-br from-red-50 via-red-100/40 to-red-50/60' : 'bg-gradient-to-br from-blue-50 via-white to-cyan-50/60 border-blue-100'
    }`}>
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-600 flex items-center justify-center shadow-sm">
            <svg viewBox="0 0 24 24" className="w-5 h-5 text-white" fill="currentColor">
              <path d="M12 2C10.9 2 10 2.9 10 4v9.17c-1.17.41-2 1.52-2 2.83 0 1.66 1.34 3 3 3 .35 0 .69-.06 1-.17.31.11.65.17 1 .17 1.66 0 3-1.34 3-3 0-1.31-.83-2.42-2-2.83V4c0-1.1-.9-2-2-2z"/>
            </svg>
          </div>
          <span className="text-sm font-semibold text-gray-700">Temperature Gauge</span>
        </div>
      </div>

      <div className="relative flex items-center justify-center">
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
          {/* Background arc */}
          <path d={describeArc(cx, cy, radius, 0, 180)} stroke="#E5F3FA" strokeWidth="20" fill="none" />
          
          {/* Safe zone - Dark Green (max to threshold - LEFT side, higher temps) */}
          <path d={describeArc(cx, cy, radius, safeStart, safeEnd)} 
                stroke="#15803d" strokeWidth="20" fill="none" />
          
          {/* Danger zone - Multi-layer for enhanced visibility */}
          <path d={describeArc(cx, cy, radius, dangerStart, dangerEnd)} 
                stroke="#b91c1c" strokeWidth="26" fill="none" opacity="1.0" />
          <path d={describeArc(cx, cy, radius, dangerStart, dangerEnd)} 
                stroke="#dc2626" strokeWidth="22" fill="none" opacity="1.0" />
          <path d={describeArc(cx, cy, radius, dangerStart, dangerEnd)} 
                stroke="#ef4444" strokeWidth="24" fill="none" opacity="0.6" />
          <path d={describeArc(cx, cy, radius, dangerStart, dangerEnd)} 
                stroke="#fca5a5" strokeWidth="32" fill="none" opacity="0.3" />

          {/* Threshold dashed radial marker */}
          <line x1={cx} y1={cy} x2={thEnd.x} y2={thEnd.y} 
                stroke="#ef4444" strokeWidth="2" strokeDasharray="4 4" />
          
          {/* Needle */}
          <line x1={cx} y1={cy} x2={needleEnd.x} y2={needleEnd.y} 
                stroke="#0ea5e9" strokeWidth="4" strokeLinecap="round"
                className="transition-all duration-300 ease-out" />
          
          {/* Center cap */}
          <circle cx={cx} cy={cy} r="8" fill="#0ea5e9" stroke="#38bdf8" strokeWidth="3" />
        </svg>
      </div>

      <div className="mt-3 flex items-center justify-between">
        <div className="flex items-baseline space-x-2">
          <span className="text-4xl font-black text-gray-900 tracking-tight">{value.toFixed(3)}</span>
          <span className="text-lg font-bold text-gray-700">°C</span>
        </div>
        <div className="text-xs text-gray-600">
          Threshold: <span className="font-semibold text-red-600">{threshold.toFixed(1)}°C</span>
        </div>
      </div>

      {/* Set Threshold Button */}
      <div className="mt-4">
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setShowThresholdInput(!showThresholdInput)}
            className="flex-1 px-4 py-2 bg-gradient-to-r from-slate-600 to-slate-700 hover:from-slate-700 hover:to-slate-800 text-white text-sm font-semibold rounded-lg shadow-sm transition-all duration-200"
          >
            {showThresholdInput ? 'Cancel' : 'Set Threshold'}
          </button>
          {exceeded && (
            <span className="flex-1 inline-flex items-center justify-center px-3 py-2 rounded-lg text-xs font-bold bg-red-600 text-white border-2 border-red-700 shadow-lg shadow-red-500/50 animate-pulse">
              🚨 DANGER: Below Threshold!
            </span>
          )}
        </div>

        {/* Threshold Input Form */}
        {showThresholdInput && (
          <div className="mt-3 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <label className="block text-xs font-semibold text-gray-700 mb-2">
              Enter New Threshold (°C)
            </label>
            <div className="flex items-center space-x-2">
              <input
                type="number"
                step="0.1"
                value={tempThreshold}
                onChange={(e) => setTempThreshold(e.target.value)}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm text-gray-900 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Enter threshold"
              />
              <button
                onClick={handleApplyThreshold}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold rounded-lg shadow-sm transition-all duration-200"
              >
                Apply
              </button>
            </div>
            <p className="mt-2 text-xs text-gray-500">
              Current: {threshold.toFixed(1)}°C
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default function TemperaturePage() {
  const [user, setUser] = useState(null)
  const router = useRouter()
  
  // Real-time data
  const [currentTemp, setCurrentTemp] = useState(44.5)
  const [timeToTarget, setTimeToTarget] = useState({ hours: 2, minutes: 45, seconds: 0 })
  const [showExcelModal, setShowExcelModal] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [userThreshold, setUserThreshold] = useState(31.7)
  const [series, setSeries] = useState({ times: [], current: [], predicted: [], threshold: 31.7 })
  const [playIndex, setPlayIndex] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [fileName, setFileName] = useState('Report_20250604-3.xls')
  const [dataVersion, setDataVersion] = useState(0)
  const fileInputRef = useRef(null)
  const hasLoadedData = useRef(false)
  
  // Alert system state
  const [alertConfig, setAlertConfig] = useState({ emails: [], leadTimeMinutes: 15, customMessage: '' })
  const [alertTriggered, setAlertTriggered] = useState(false)
  const alertSentRef = useRef(false)
  const dangerAlertSentRef = useRef(false)
  
  // Tab state
  const [activeTab, setActiveTab] = useState('live')
  const [hasUploadedFile, setHasUploadedFile] = useState(false)
  
  // Upload preview states
  const [showUploadPreview, setShowUploadPreview] = useState(false)
  const [previewData, setPreviewData] = useState(null)
  
  // Success notification state
  const [showNotification, setShowNotification] = useState(false)

  // Parse CSV text
  const parseCsv = (text) => {
    const [header, ...lines] = text.trim().split(/\r?\n/)
    const cols = header.split(',').map(h => h.trim().toLowerCase())
    const idx = {
      timestamp: cols.indexOf('timestamp'),
      current: cols.indexOf('current'),
      predicted: cols.indexOf('predicted'),
      threshold: cols.indexOf('threshold')
    }
    const rows = lines.map(line => {
      const c = line.split(',')
      return {
        timestamp: idx.timestamp >= 0 ? c[idx.timestamp] : undefined,
        current: idx.current >= 0 ? parseFloat(c[idx.current]) : undefined,
        predicted: idx.predicted >= 0 ? parseFloat(c[idx.predicted]) : undefined,
        threshold: idx.threshold >= 0 ? parseFloat(c[idx.threshold]) : undefined,
      }
    })
    return rows
  }

  // Parse XLSX ArrayBuffer
  const parseXlsx = async (buffer) => {
    try {
      const XLSX = await import('xlsx')
      const wb = XLSX.read(buffer, { type: 'array' })
      const sheetName = wb.SheetNames[0]
      const ws = wb.Sheets[sheetName]
      const json = XLSX.utils.sheet_to_json(ws, { defval: '' })
        
      return json.map(row => {
        const keys = Object.keys(row)
        const timeKey = keys.find(k => k.toLowerCase().includes('date') || k.toLowerCase().includes('time'))
        const tempKey = keys.find(k => k.toLowerCase() === 'temp' || k.toLowerCase() === 'temperature' || k.toLowerCase() === 'current')
        const predKey = keys.find(k => k.toLowerCase() === 'predicted')
        const threshKey = keys.find(k => k.toLowerCase() === 'threshold')
          
        let timestamp = timeKey ? row[timeKey] : undefined
        if (typeof timestamp === 'number') {
          const fractionalDay = timestamp - Math.floor(timestamp)
          const totalMinutes = Math.round(fractionalDay * 24 * 60)
          const hours = Math.floor(totalMinutes / 60)
          const minutes = totalMinutes % 60
          timestamp = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`
        }
          
        return {
          timestamp,
          current: tempKey ? parseFloat(row[tempKey]) : undefined,
          predicted: predKey ? parseFloat(row[predKey]) : undefined,
          threshold: threshKey ? parseFloat(row[threshKey]) : undefined,
        }
      })
    } catch (e) {
      console.error('XLSX parsing error:', e)
      alert('Error parsing XLSX file: ' + e.message)
      return []
    }
  }

  // Handle file upload - ASYNC function to use await
  const handleFileUpload = async (file) => {
    if (!file) {
      console.log('No file selected')
      return
    }
    
    console.log('✅ File selected:', file.name, 'Type:', file.type, 'Size:', file.size, 'bytes')

    const fileName = file.name.toLowerCase()
    console.log('Parsing file:', fileName)
    
    if (fileName.endsWith('.csv')) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const rows = parseCsv(e.target.result)
        const times = rows.map(r => r.timestamp)
        const current = rows.map(r => r.current).filter(v => typeof v === 'number')
        const predicted = rows.map(r => r.predicted ?? undefined).filter(v => typeof v === 'number')
        const threshold = rows.find(r => typeof r.threshold === 'number')?.threshold ?? 31.7
        
        console.log('CSV parsed:', { times: times.length, current: current.length })
        
        setPreviewData({
          fileName: file.name,
          times,
          current,
          predicted,
          threshold,
          stats: {
            dataPoints: current.length,
            min: Math.min(...current),
            max: Math.max(...current),
            avg: (current.reduce((a,b) => a+b, 0) / current.length),
            timeRange: times.length > 0 ? `${times[0]} - ${times[times.length - 1]}` : 'N/A'
          }
        })
        setShowUploadPreview(true)
        
        if (fileInputRef.current) {
          fileInputRef.current.value = ''
        }
      }
      reader.readAsText(file)
    } else if (fileName.endsWith('.xlsx') || fileName.endsWith('.xls')) {
      try {
        const buffer = await file.arrayBuffer()
        const rows = await parseXlsx(buffer)
        const times = rows.map(r => r.timestamp).filter(t => t !== undefined)
        const current = rows.map(r => r.current).filter(v => typeof v === 'number')
        const predicted = rows.map(r => r.predicted ?? undefined).filter(v => typeof v === 'number')
        const threshold = rows.find(r => typeof r.threshold === 'number')?.threshold ?? 31.7
        
        console.log('Excel parsed:', { times: times.length, current: current.length })
        
        setPreviewData({
          fileName: file.name,
          times,
          current,
          predicted,
          threshold,
          stats: {
            dataPoints: current.length,
            min: Math.min(...current),
            max: Math.max(...current),
            avg: (current.reduce((a,b) => a+b, 0) / current.length),
            timeRange: times.length > 0 ? `${times[0]} - ${times[times.length - 1]}` : 'N/A'
          }
        })
        setShowUploadPreview(true)
        
        if (fileInputRef.current) {
          fileInputRef.current.value = ''
        }
      } catch (error) {
        console.error('Error parsing Excel file:', error)
      }
    } else {
      console.log('Unsupported file type')
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <TopNav user={user} />
        <main className="flex-1 overflow-y-auto p-8">
          <div className="max-w-7xl mx-auto">
            <h1 className="text-3xl font-bold text-gray-900 mb-6">Temperature Monitoring</h1>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <TemperatureGauge
                value={currentTemp}
                threshold={series.threshold}
                onThresholdChange={(newThreshold) => {
                  setUserThreshold(newThreshold)
                  setSeries(prev => ({ ...prev, threshold: newThreshold }))
                }}
              />
              
              <div className="bg-white rounded-2xl p-6 shadow-xl border border-gray-100">
                <h2 className="text-xl font-semibold text-gray-800 mb-4">Upload Data</h2>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  onChange={(e) => handleFileUpload(e.target.files[0])}
                  className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                />
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
