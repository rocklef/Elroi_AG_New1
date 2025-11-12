'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { supabase } from '@/lib/supabase'
import Sidebar from '@/components/Sidebar'

export default function TemperaturePage() {
  const [user, setUser] = useState(null)
  const router = useRouter()
  const [cardAnimationClasses, setCardAnimationClasses] = useState([
    'opacity-0 translate-y-4', 'opacity-0 translate-y-4', 'opacity-0 translate-y-4', 'opacity-0 translate-y-4'
  ])
  
  // Real-time data
  const [currentTemp, setCurrentTemp] = useState(44.5)
  const [timeToTarget, setTimeToTarget] = useState({ hours: 2, minutes: 45, seconds: 0 })
  const [showExcelModal, setShowExcelModal] = useState(false)

  // Format temperature for display (e.g., 44.5 -> ["4", "4"])
  const getTempDigits = (temp) => {
    const tempStr = Math.floor(temp).toString().padStart(2, '0')
    return tempStr.split('')
  }

  // Format time for display
  const getTimeDigits = () => {
    const minutes = timeToTarget.minutes.toString().padStart(2, '0')
    const seconds = timeToTarget.seconds.toString().padStart(2, '0')
    return { minutes: minutes.split(''), seconds: seconds.split('') }
  }

  useEffect(() => {
    const checkUser = async () => {
      const { data: { session } } = await supabase.auth.getSession()
      if (!session) {
        router.push('/login')
      } else {
        setUser(session.user)
      }
    }

    checkUser()

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!session) {
        router.push('/login')
      } else {
        setUser(session.user)
      }
    })

    // Staggered card animations
    const timers = [100, 200, 300, 400].map((delay, index) => {
      return setTimeout(() => {
        setCardAnimationClasses(prev => {
          const newClasses = [...prev]
          newClasses[index] = 'opacity-100 translate-y-0'
          return newClasses
        })
      }, delay)
    })

    // Real-time updates - count up timer and temperature simulation
    const updateInterval = setInterval(() => {
      setTimeToTarget(prev => {
        let { hours, minutes, seconds } = prev
        
        // Count UP logic (time elapsed)
        seconds++
        if (seconds >= 60) {
          seconds = 0
          minutes++
        }
        if (minutes >= 60) {
          minutes = 0
          hours++
        }
        
        return { hours, minutes, seconds }
      })
      
      // Simulate temperature decreasing slowly (cooling effect)
      setCurrentTemp(prev => {
        const newTemp = prev - 0.02 // Decrease by 0.02°C per second
        return Math.max(35.0, parseFloat(newTemp.toFixed(2))) // Don't go below target temp
      })
    }, 1000)

    return () => {
      subscription.unsubscribe()
      timers.forEach(timer => clearTimeout(timer))
      clearInterval(updateInterval)
    }
  }, [router])

  if (!user) {
    return <div className="min-h-screen flex items-center justify-center bg-white">Loading...</div>
  }

  // Top 4 Metric Cards
  const metricCards = [
    {
      id: 'current',
      title: 'Current Temperature',
      value: '44.5',
      unit: '°C',
      subtext: 'Δ +0.8°C since last update',
      bgColor: 'bg-[#0071CE]',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      )
    },
    {
      id: 'target',
      title: 'Target Temperature',
      value: '35.0',
      unit: '°C',
      subtext: 'Stabilization ETA 2h 45m',
      bgColor: 'bg-[#FF6B35]',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      )
    },
    {
      id: 'threshold',
      title: 'Threshold Temperature',
      value: '50.0',
      unit: '°C',
      subtext: 'Status: Above limit ⚠',
      bgColor: 'bg-[#E94E4E]',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      )
    },
    {
      id: 'time',
      title: 'Time to Target',
      value: '2:45:00',
      unit: '',
      subtext: 'Confidence 94%',
      bgColor: 'bg-[#00D9C0]',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      )
    }
  ]

  return (
    <div className="flex min-h-screen bg-white">
      <Sidebar activeSection="temperature" />

      <div className="flex-1 flex flex-col">
        {/* Top Navigation Bar */}
        <header className="bg-[#081440] h-[70px] border-b border-gray-200">
          <div className="flex items-center justify-between px-8 h-full">
            <h1 className="text-white text-xl font-semibold">Temperature Analysis</h1>
            
            <div className="flex items-center space-x-6">
              <div className="flex items-center text-sm text-gray-300">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>12:47 AM</span>
              </div>
              <button className="relative p-2 text-gray-300 hover:text-white transition-colors cursor-pointer">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                </svg>
                <span className="absolute top-0 right-0 block h-2 w-2 rounded-full bg-red-500 animate-pulse"></span>
              </button>
              <div className="w-8 h-8 rounded-full bg-white border border-[#0071CE] flex items-center justify-center cursor-pointer hover:scale-105 transition-transform duration-200">
                <span className="text-[#0071CE] text-sm font-semibold">JD</span>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 p-8 overflow-y-auto">
          <div className="max-w-7xl mx-auto">
            {/* Section Header */}
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-[#0B0B0B] mb-1">Temperature Monitoring Dashboard</h2>
              <p className="text-[#5B6C84] text-sm">Real-time temperature analytics and predictive insights</p>
            </div>

            {/* Top 4 Metric Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              {metricCards.map((card, index) => (
                <div
                  key={card.id}
                  className={`${card.bgColor} rounded-xl p-6 cursor-pointer transform transition-all duration-300 ease-out hover:scale-105 ${cardAnimationClasses[index]}`}
                  style={{ 
                    borderRadius: '14px', 
                    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.06)',
                    minHeight: '150px'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.08)'
                    e.currentTarget.style.transform = 'translateY(-4px) scale(1.02)'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.06)'
                    e.currentTarget.style.transform = 'translateY(0) scale(1)'
                  }}
                >
                  <h3 className="text-white text-sm font-medium mb-3">{card.title}</h3>
                  <p className="text-white text-3xl font-bold mb-1">
                    {card.value}<span className="text-xl ml-1">{card.unit}</span>
                  </p>
                  <p className="text-white text-xs opacity-80">{card.subtext}</p>
                </div>
              ))}
            </div>

            {/* Main Split Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              {/* Left: Graph Panel */}
              <div className="bg-white rounded-xl p-6 border-2 border-[#0071CE] shadow-lg hover:shadow-2xl transition-shadow duration-300" style={{ boxShadow: '0 8px 24px rgba(0, 113, 206, 0.15)' }}>
                <div className="mb-4">
                  <h2 className="text-[#0B0B0B] text-lg font-bold mb-1">Temperature Profile — Actual vs Predicted</h2>
                  <p className="text-[#5B6C84] text-sm">Cycle #146 | Updated 12:47 AM</p>
                </div>

                {/* Chart Container */}
                <div className="bg-white rounded-lg h-80 border border-[#E0E3EA] p-4">
                  <div className="flex justify-end mb-2 space-x-4 text-xs">
                    <div className="flex items-center">
                      <div className="w-3 h-0.5 bg-[#0071CE] mr-1"></div>
                      <span className="text-[#5B6C84]">Current Data</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-0.5 bg-[#00D9C0] mr-1" style={{ backgroundImage: 'repeating-linear-gradient(to right, #00D9C0 0px, #00D9C0 4px, transparent 4px, transparent 8px)' }}></div>
                      <span className="text-[#5B6C84]">Predicted Cooling</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-0.5 bg-[#FF6B35] mr-1" style={{ backgroundImage: 'repeating-linear-gradient(to right, #FF6B35 0px, #FF6B35 4px, transparent 4px, transparent 8px)' }}></div>
                      <span className="text-[#5B6C84]">Threshold</span>
                    </div>
                  </div>

                  {/* Chart with realistic cooling curve */}
                  <div className="relative w-full h-64">
                    {/* Y-axis with values */}
                    <div className="absolute left-0 top-0 bottom-8 flex flex-col justify-between w-12">
                      {[65, 60, 55, 50, 45, 40, 35].map((temp, i) => (
                        <div key={i} className="flex items-center justify-end">
                          <span className="text-xs text-[#5B6C84] font-medium">{temp}</span>
                        </div>
                      ))}
                    </div>

                    {/* Grid lines */}
                    <div className="absolute left-12 right-0 top-0 bottom-8 flex flex-col justify-between">
                      {[0, 1, 2, 3, 4, 5, 6].map((i) => (
                        <div key={i} className="border-t border-[#E0E3EA]"></div>
                      ))}
                    </div>
                    
                    {/* Threshold Line (Red Dashed at 35°C) */}
                    <svg className="absolute left-12 top-0 w-[calc(100%-3rem)] h-[calc(100%-2rem)]" preserveAspectRatio="none">
                      <line
                        x1="0"
                        y1="85%"
                        x2="100%"
                        y2="85%"
                        stroke="#FF6B35"
                        strokeWidth="2"
                        strokeDasharray="8,4"
                        opacity="0.7"
                      />
                    </svg>
                    
                    {/* Current Data Line (Blue Solid - Cooling Curve) */}
                    <svg className="absolute left-12 top-0 w-[calc(100%-3rem)] h-[calc(100%-2rem)]" preserveAspectRatio="none">
                      <path
                        d="M 0,8 Q 15,12 30,18 T 60,28 T 90,35 T 120,40 T 150,43 T 180,45 T 200,46"
                        fill="none"
                        stroke="#0071CE"
                        strokeWidth="3"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        style={{
                          filter: 'drop-shadow(0 2px 4px rgba(0, 113, 206, 0.3))'
                        }}
                      />
                    </svg>
                    
                    {/* Predicted Cooling Line (Teal Dotted) */}
                    <svg className="absolute left-12 top-0 w-[calc(100%-3rem)] h-[calc(100%-2rem)]" preserveAspectRatio="none">
                      <path
                        d="M 200,46 Q 230,48 260,52 T 320,60 T 380,67 T 440,73 T 500,78 T 560,82 T 600,85"
                        fill="none"
                        stroke="#00D9C0"
                        strokeWidth="3"
                        strokeDasharray="6,6"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        opacity="0.85"
                      />
                    </svg>

                    {/* Y-axis label */}
                    <div className="absolute left-0 top-1/2 -translate-y-1/2 -rotate-90 origin-center">
                      <span className="text-xs text-[#5B6C84] font-semibold whitespace-nowrap">Temperature (°C)</span>
                    </div>
                  </div>

                  {/* X-Axis Labels with realistic times */}
                  <div className="flex justify-between text-xs text-[#5B6C84] mt-2 pl-12 pr-0">
                    <span>21:30</span>
                    <span>22:00</span>
                    <span>22:30</span>
                    <span>23:00</span>
                    <span>23:30</span>
                    <span>00:00</span>
                  </div>
                  
                  {/* X-axis label */}
                  <div className="text-center mt-1">
                    <span className="text-xs text-[#5B6C84] font-semibold">Time</span>
                  </div>
                </div>
              </div>

              {/* Right: Digital Display Panel */}
              <div className="bg-white rounded-xl p-6 border-2 border-[#00D9C0] shadow-lg hover:shadow-2xl transition-shadow duration-300" style={{ boxShadow: '0 8px 24px rgba(0, 217, 192, 0.15)' }}>
                <h2 className="text-[#0B0B0B] text-lg font-bold mb-6">Live Temperature Reading</h2>
                
                {/* Combined Digital Display - Time and Temperature Side by Side */}
                <div className="flex justify-center mb-8">
                  <div className="bg-white rounded-xl p-5 shadow-lg border-2 border-[#00D9C0]">
                    <div className="flex items-center justify-center space-x-6">
                      {/* LEFT SIDE: Minutes and Seconds */}
                      <div className="flex flex-col items-center">
                        <div className="flex items-center space-x-2 mb-2">
                          {/* Minutes */}
                          <div className="flex space-x-1.5">
                            <div className="bg-gray-100 rounded-md w-12 h-14 flex items-center justify-center shadow-sm border border-gray-200">
                              <span className="text-3xl font-bold text-gray-900">{getTimeDigits().minutes[0]}</span>
                            </div>
                            <div className="bg-gray-100 rounded-md w-12 h-14 flex items-center justify-center shadow-sm border border-gray-200">
                              <span className="text-3xl font-bold text-gray-900">{getTimeDigits().minutes[1]}</span>
                            </div>
                          </div>
                          
                          {/* Separator Dots */}
                          <div className="flex flex-col space-y-1.5 pb-1 bg-gray-100 rounded px-1.5 py-2">
                            <div className="w-2.5 h-2.5 rounded-full bg-[#0071CE]"></div>
                            <div className="w-2.5 h-2.5 rounded-full bg-[#0071CE]"></div>
                          </div>
                          
                          {/* Seconds */}
                          <div className="flex space-x-1.5">
                            <div className="bg-gray-100 rounded-md w-12 h-14 flex items-center justify-center shadow-sm border border-gray-200">
                              <span className="text-3xl font-bold text-gray-900">{getTimeDigits().seconds[0]}</span>
                            </div>
                            <div className="bg-gray-100 rounded-md w-12 h-14 flex items-center justify-center shadow-sm border border-gray-200">
                              <span className="text-3xl font-bold text-gray-900">{getTimeDigits().seconds[1]}</span>
                            </div>
                          </div>
                        </div>
                        <div className="flex justify-around w-full space-x-8">
                          <span className="text-[10px] text-[#5B6C84] font-semibold">MINUTES</span>
                          <span className="text-[10px] text-[#5B6C84] font-semibold">SECONDS</span>
                        </div>
                      </div>

                      {/* Vertical Divider */}
                      <div className="w-px h-16 bg-[#E0E3EA]"></div>

                      {/* RIGHT SIDE: Current Temperature */}
                      <div className="flex flex-col items-center">
                        <div className="flex items-center space-x-1.5 mb-2">
                          {/* Temperature Digits */}
                          <div className="flex space-x-1.5">
                            {getTempDigits(currentTemp).map((digit, idx) => (
                              <div key={idx} className="bg-gray-100 rounded-md w-12 h-14 flex items-center justify-center shadow-sm border border-gray-200">
                                <span className="text-3xl font-bold text-[#0071CE]">{digit}</span>
                              </div>
                            ))}
                          </div>
                          {/* Decimal and Unit */}
                          <div className="flex flex-col items-start justify-center bg-gray-100 rounded-md px-2 py-1">
                            <span className="text-2xl font-bold text-[#0071CE]">.{Math.round((currentTemp % 1) * 10)}</span>
                            <span className="text-lg font-semibold text-[#5B6C84]">°C</span>
                          </div>
                        </div>
                        <span className="text-[10px] text-[#5B6C84] font-semibold">CURRENT TEMP</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Three Mini Info Cards */}
                <div className="grid grid-cols-1 gap-3">
                  <div className="bg-white border-2 border-[#0071CE] rounded-lg p-3 flex items-center space-x-3 hover:border-[#005BA3] hover:bg-blue-50 transition-all duration-200 shadow-md" style={{ boxShadow: '0 4px 12px rgba(0, 113, 206, 0.1)' }}>
                    <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-[#0071CE]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                      </svg>
                    </div>
                    <div>
                      <p className="text-xs text-[#5B6C84]">Δ Since Last Update</p>
                      <p className="text-sm font-bold text-[#0B0B0B]">+0.8 °C</p>
                    </div>
                  </div>

                  <div className="bg-white border-2 border-[#FF6B35] rounded-lg p-3 flex items-center space-x-3 hover:border-[#E85A24] hover:bg-orange-50 transition-all duration-200 shadow-md" style={{ boxShadow: '0 4px 12px rgba(255, 107, 53, 0.1)' }}>
                    <div className="w-8 h-8 rounded-full bg-orange-100 flex items-center justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-[#FF6B35]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <div>
                      <p className="text-xs text-[#5B6C84]">Stabilization ETA</p>
                      <p className="text-sm font-bold text-[#0B0B0B]">≈ 2 h 45 m</p>
                    </div>
                  </div>

                  <div className="bg-white border-2 border-[#00D9C0] rounded-lg p-3 flex items-center space-x-3 hover:border-[#00B89F] hover:bg-teal-50 transition-all duration-200 shadow-md" style={{ boxShadow: '0 4px 12px rgba(0, 217, 192, 0.1)' }}>
                    <div className="w-8 h-8 rounded-full bg-teal-100 flex items-center justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-[#00D9C0]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <div>
                      <p className="text-xs text-[#5B6C84]">Model Confidence</p>
                      <p className="text-sm font-bold text-[#0B0B0B]">94 %</p>
                    </div>
                  </div>
                </div>

                {/* View Full Excel Button */}
                <div className="mt-6 flex justify-center">
                  <button 
                    onClick={() => setShowExcelModal(true)}
                    className="cursor-pointer bg-[#0071CE] hover:bg-[#005BA3] text-white font-semibold py-3 px-8 rounded-lg shadow-md transition-all duration-200 transform hover:scale-105 flex items-center space-x-2"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <span>View Full Excel Report</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Bottom Section - Performance Insights */}
            <div className="bg-white rounded-xl p-6 border-2 border-[#7B68EE] shadow-lg" style={{ boxShadow: '0 8px 24px rgba(123, 104, 238, 0.15)' }}>
              <h2 className="text-[#0B0B0B] text-xl font-bold mb-6">Performance Insights</h2>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Left: Model Accuracy Chart */}
                <div className="bg-gradient-to-br from-blue-50 to-white rounded-lg p-6 border-2 border-[#0071CE] shadow-md" style={{ boxShadow: '0 6px 16px rgba(0, 113, 206, 0.12)' }}>
                  <h3 className="text-[#0B0B0B] text-sm font-semibold mb-4">Model Accuracy Over Time</h3>
                  <div className="h-32 flex items-end justify-between space-x-2">
                    <div className="flex-1 bg-gradient-to-t from-[#0071CE] to-[#00D9C0] rounded-t" style={{ height: '70%' }}>
                      <p className="text-xs text-white text-center mt-1">92%</p>
                    </div>
                    <div className="flex-1 bg-gradient-to-t from-[#0071CE] to-[#00D9C0] rounded-t" style={{ height: '85%' }}>
                      <p className="text-xs text-white text-center mt-1">94%</p>
                    </div>
                    <div className="flex-1 bg-gradient-to-t from-[#0071CE] to-[#00D9C0] rounded-t" style={{ height: '95%' }}>
                      <p className="text-xs text-white text-center mt-1">95%</p>
                    </div>
                  </div>
                  <div className="flex justify-between mt-2 text-xs text-[#5B6C84]">
                    <span>Last Hour</span>
                    <span>Now</span>
                    <span>Predicted</span>
                  </div>
                </div>

                {/* Right: Text Summary */}
                <div className="bg-gradient-to-br from-teal-50 to-white rounded-lg p-6 border-2 border-[#00D9C0] shadow-md" style={{ boxShadow: '0 6px 16px rgba(0, 217, 192, 0.12)' }}>
                  <h3 className="text-[#0B0B0B] text-base font-bold mb-4">Automated Analysis</h3>
                  <div className="space-y-3 text-sm leading-relaxed">
                    <p className="text-[#1A1A1A] font-medium">
                      The cooling curve exhibits a <span className="font-bold text-[#0071CE] bg-blue-100 px-1 rounded">smooth exponential decay</span>.
                    </p>
                    <p className="text-[#1A1A1A] font-medium">
                      Model forecasts stabilization near <span className="font-bold text-[#FF6B35] bg-orange-100 px-1 rounded">35 °C</span> within 2 hours.
                    </p>
                    <p className="text-[#1A1A1A] font-medium">
                      Current deviation &lt; ±1.5 °C — system performing <span className="font-bold text-[#00D9C0] bg-teal-100 px-1 rounded">within tolerance</span>.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* Excel Data Modal */}
      {showExcelModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4" onClick={() => setShowExcelModal(false)}>
          <div className="bg-white rounded-2xl shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-hidden" onClick={(e) => e.stopPropagation()}>
            {/* Modal Header */}
            <div className="bg-gradient-to-r from-[#0071CE] to-[#00D9C0] p-6 flex justify-between items-center">
              <h2 className="text-white text-2xl font-bold">Temperature Analysis Report</h2>
              <button 
                onClick={() => setShowExcelModal(false)}
                className="text-white hover:bg-white hover:bg-opacity-20 rounded-full p-2 transition-all duration-200"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Modal Body - Excel-like Table */}
            <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="bg-[#0071CE] text-white">
                      <th className="border border-gray-300 px-4 py-3 text-left font-semibold">Timestamp</th>
                      <th className="border border-gray-300 px-4 py-3 text-left font-semibold">Current Temp (°C)</th>
                      <th className="border border-gray-300 px-4 py-3 text-left font-semibold">Target Temp (°C)</th>
                      <th className="border border-gray-300 px-4 py-3 text-left font-semibold">Threshold (°C)</th>
                      <th className="border border-gray-300 px-4 py-3 text-left font-semibold">Deviation</th>
                      <th className="border border-gray-300 px-4 py-3 text-left font-semibold">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { time: '12:47:00', current: '44.5', target: '35.0', threshold: '50.0', deviation: '+0.8', status: 'Normal' },
                      { time: '12:46:00', current: '43.7', target: '35.0', threshold: '50.0', deviation: '+0.5', status: 'Normal' },
                      { time: '12:45:00', current: '43.2', target: '35.0', threshold: '50.0', deviation: '+0.3', status: 'Normal' },
                      { time: '12:44:00', current: '42.9', target: '35.0', threshold: '50.0', deviation: '+0.2', status: 'Normal' },
                      { time: '12:43:00', current: '42.7', target: '35.0', threshold: '50.0', deviation: '-0.1', status: 'Normal' },
                      { time: '12:42:00', current: '42.8', target: '35.0', threshold: '50.0', deviation: '+0.4', status: 'Normal' },
                      { time: '12:41:00', current: '42.4', target: '35.0', threshold: '50.0', deviation: '+0.6', status: 'Normal' },
                      { time: '12:40:00', current: '41.8', target: '35.0', threshold: '50.0', deviation: '+0.9', status: 'Normal' },
                      { time: '12:39:00', current: '40.9', target: '35.0', threshold: '50.0', deviation: '+1.2', status: 'Normal' },
                      { time: '12:38:00', current: '39.7', target: '35.0', threshold: '50.0', deviation: '+1.5', status: 'Normal' },
                      { time: '12:37:00', current: '38.2', target: '35.0', threshold: '50.0', deviation: '+0.8', status: 'Normal' },
                      { time: '12:36:00', current: '37.4', target: '35.0', threshold: '50.0', deviation: '+0.3', status: 'Normal' },
                    ].map((row, idx) => (
                      <tr key={idx} className={idx % 2 === 0 ? 'bg-gray-50' : 'bg-white'} >
                        <td className="border border-gray-300 px-4 py-2 text-[#0B0B0B]">{row.time}</td>
                        <td className="border border-gray-300 px-4 py-2 text-[#0071CE] font-semibold">{row.current}</td>
                        <td className="border border-gray-300 px-4 py-2 text-[#FF6B35] font-semibold">{row.target}</td>
                        <td className="border border-gray-300 px-4 py-2 text-[#E94E4E] font-semibold">{row.threshold}</td>
                        <td className="border border-gray-300 px-4 py-2 text-[#0B0B0B]">{row.deviation}</td>
                        <td className="border border-gray-300 px-4 py-2">
                          <span className="bg-[#00D9C0] text-white px-3 py-1 rounded-full text-xs font-semibold">{row.status}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Export Button */}
              <div className="mt-6 flex justify-end space-x-3">
                <button className="cursor-pointer bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold py-2 px-6 rounded-lg transition-colors duration-200">
                  Print Report
                </button>
                <button className="cursor-pointer bg-[#00D9C0] hover:bg-[#00B89F] text-white font-semibold py-2 px-6 rounded-lg transition-colors duration-200 flex items-center space-x-2">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  <span>Download Excel</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
