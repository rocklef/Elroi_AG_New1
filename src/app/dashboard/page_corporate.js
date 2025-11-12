'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { supabase } from '@/lib/supabase'
import Sidebar from '@/components/Sidebar'

export default function DashboardPage() {
  const [user, setUser] = useState(null)
  const router = useRouter()
  const [currentTime, setCurrentTime] = useState('')
  const [cardAnimationClasses, setCardAnimationClasses] = useState([
    'opacity-0 translate-y-2', 'opacity-0 translate-y-2', 'opacity-0 translate-y-2', 
    'opacity-0 translate-y-2', 'opacity-0 translate-y-2'
  ])

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

    // Update time
    const updateTime = () => {
      const now = new Date()
      setCurrentTime(now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' }))
    }
    updateTime()
    const timeInterval = setInterval(updateTime, 1000)

    // Auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!session) {
        router.push('/login')
      } else {
        setUser(session.user)
      }
    })

    // Staggered card animations (0.1s delay)
    const timers = [100, 200, 300, 400, 500].map((delay, index) => {
      return setTimeout(() => {
        setCardAnimationClasses(prev => {
          const newClasses = [...prev]
          newClasses[index] = 'opacity-100 translate-y-0'
          return newClasses
        })
      }, delay)
    })

    return () => {
      subscription.unsubscribe()
      clearInterval(timeInterval)
      timers.forEach(timer => clearTimeout(timer))
    }
  }, [router])

  if (!user) {
    return <div className="min-h-screen flex items-center justify-center">Loading...</div>
  }

  // 5 Parameters - Professional corporate style with left accent borders
  const parameters = [
    {
      title: 'Temperature',
      value: '44.5',
      unit: '°C',
      subtext: 'Δ +0.8°C since last update',
      confidence: 'Model confidence 94%',
      accentColor: '#0A5AD8',
      trend: 'up',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
      isPrimary: true
    },
    {
      title: 'Pressure',
      value: '2.1',
      unit: ' bar',
      subtext: 'Δ -0.2 predicted',
      confidence: 'Accuracy 96%',
      accentColor: '#0056A1',
      trend: 'down',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      )
    },
    {
      title: 'Humidity',
      value: '58',
      unit: '%',
      subtext: 'trending upward',
      confidence: 'Model confidence 92%',
      accentColor: '#007C89',
      trend: 'up',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
        </svg>
      )
    },
    {
      title: 'Vibration',
      value: '2.3',
      unit: ' mm/s',
      subtext: 'RMS velocity',
      confidence: 'Accuracy 89%',
      accentColor: '#4B5ACF',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      )
    },
    {
      title: 'Add Parameter',
      value: '+',
      unit: '',
      subtext: 'Configure new sensor',
      confidence: '',
      accentColor: '#B0BEC5',
      isDashed: true,
      icon: null
    }
  ]

  return (
    <div className="flex min-h-screen" style={{ background: '#F7F9FC' }}>
      <Sidebar activeSection="dashboard" />

      <div className="flex-1 flex flex-col">
        {/* Solid Navy Top Bar */}
        <header 
          style={{ 
            background: '#081440',
            borderBottom: '1px solid rgba(255, 255, 255, 0.15)'
          }}
        >
          <div className="flex items-center justify-between px-8 h-16">
            <div className="flex items-center space-x-3">
              <div 
                className="w-8 h-8 rounded flex items-center justify-center"
                style={{ background: '#0A5AD8' }}
              >
                <span className="text-white font-bold text-sm" style={{ fontFamily: 'Inter, sans-serif' }}>E</span>
              </div>
              <h1 
                className="text-white text-base font-semibold"
                style={{ fontFamily: 'Inter, sans-serif', fontWeight: 600 }}
              >
                Predictive Maintenance | Live Overview
              </h1>
            </div>
            <div className="flex items-center space-x-6">
              <div 
                className="flex items-center text-sm text-white/90 font-medium"
                style={{ fontFamily: 'Inter, sans-serif' }}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>{currentTime}</span>
              </div>
              <button className="relative p-2 text-white/80 hover:text-white transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                </svg>
                <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
              </button>
              <div 
                className="w-9 h-9 rounded-full flex items-center justify-center cursor-pointer hover:opacity-90 transition-opacity"
                style={{ 
                  background: 'white',
                  border: '2px solid white'
                }}
              >
                <span className="text-[#081440] text-xs font-semibold" style={{ fontFamily: 'Inter, sans-serif' }}>JD</span>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 p-8">
          {/* Section Title */}
          <div className="mb-6">
            <h2 
              className="text-lg font-bold mb-1"
              style={{ color: '#1A1A1A', fontFamily: 'Inter, sans-serif', fontWeight: 700 }}
            >
              System Overview
            </h2>
            <p 
              className="text-sm"
              style={{ color: '#5B6C84', fontFamily: 'Inter, sans-serif', fontWeight: 500 }}
            >
              TCN Model v2.3 (Active)
            </p>
          </div>

          {/* Centered Container ~90% width */}
          <div className="max-w-7xl">
            {/* 2x3 Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-7 gap-y-8">
              {parameters.map((param, index) => (
                <div
                  key={index}
                  className={`rounded-xl p-6 cursor-pointer transition-all duration-250 ease-out ${
                    param.isDashed
                      ? 'border-2 border-dashed hover:border-[#0A5AD8]'
                      : 'hover:-translate-y-0.5'
                  } ${cardAnimationClasses[index]}`}
                  style={{
                    background: 'white',
                    boxShadow: '0 2px 12px rgba(0, 0, 0, 0.06)',
                    border: param.isDashed ? '2px dashed #B0BEC5' : '1px solid #E1E4EB',
                    borderLeft: param.isDashed ? '2px dashed #B0BEC5' : `4px solid ${param.accentColor}`,
                    borderRadius: '12px',
                    minHeight: param.isPrimary ? '200px' : '180px'
                  }}
                  onMouseEnter={(e) => {
                    if (!param.isDashed) {
                      e.currentTarget.style.boxShadow = `0 4px 16px rgba(10, 90, 216, 0.12)`
                      e.currentTarget.style.borderColor = '#0A5AD8'
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!param.isDashed) {
                      e.currentTarget.style.boxShadow = '0 2px 12px rgba(0, 0, 0, 0.06)'
                      e.currentTarget.style.borderColor = '#E1E4EB'
                    }
                  }}
                >
                  {param.isDashed ? (
                    <div className="flex flex-col items-center justify-center h-full min-h-[160px]">
                      <div 
                        className="w-12 h-12 rounded-full flex items-center justify-center mb-3 transition-all duration-200 hover:scale-105"
                        style={{ background: '#F7F9FC' }}
                      >
                        <span className="text-2xl font-light" style={{ color: '#5B6C84' }}>+</span>
                      </div>
                      <h3 
                        className="text-sm font-semibold mb-1"
                        style={{ color: '#1A1A1A', fontFamily: 'Inter, sans-serif', fontWeight: 600 }}
                      >
                        {param.title}
                      </h3>
                      <p className="text-xs" style={{ color: '#5B6C84', fontFamily: 'Inter, sans-serif' }}>
                        {param.subtext}
                      </p>
                    </div>
                  ) : (
                    <div>
                      {/* Top row - icon + title */}
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-2">
                          <div 
                            className="w-9 h-9 rounded-lg flex items-center justify-center"
                            style={{ background: `${param.accentColor}15`, color: param.accentColor }}
                          >
                            {param.icon}
                          </div>
                          <h3 
                            className="text-sm font-bold uppercase tracking-wide"
                            style={{ color: '#1A1A1A', fontFamily: 'Inter, sans-serif', fontWeight: 700 }}
                          >
                            {param.title}
                          </h3>
                        </div>
                        {param.trend && (
                          <svg 
                            xmlns="http://www.w3.org/2000/svg" 
                            className="h-4 w-4" 
                            fill="none" 
                            viewBox="0 0 24 24" 
                            stroke="currentColor"
                            style={{ 
                              color: param.trend === 'up' ? '#2E7D32' : '#D32F2F',
                              transform: param.trend === 'down' ? 'rotate(180deg)' : 'none'
                            }}
                          >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                          </svg>
                        )}
                      </div>
                      
                      {/* Large metric value */}
                      <p 
                        className="mb-3"
                        style={{ 
                          color: '#0A5AD8', 
                          fontFamily: 'Poppins, sans-serif', 
                          fontSize: param.isPrimary ? '42px' : '38px',
                          fontWeight: 600,
                          lineHeight: 1
                        }}
                      >
                        {param.value}<span style={{ fontSize: param.isPrimary ? '28px' : '24px' }}>{param.unit}</span>
                      </p>
                      
                      {/* Subtext */}
                      <p 
                        className="text-xs mb-1"
                        style={{ color: '#5B6C84', fontFamily: 'Inter, sans-serif', fontWeight: 500 }}
                      >
                        {param.subtext}
                      </p>
                      
                      {/* Confidence */}
                      {param.confidence && (
                        <p 
                          className="text-xs"
                          style={{ color: '#5B6C84', fontFamily: 'Inter, sans-serif', fontWeight: 400 }}
                        >
                          {param.confidence}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
