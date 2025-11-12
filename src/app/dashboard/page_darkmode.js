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
    'opacity-0 translate-y-3', 'opacity-0 translate-y-3', 'opacity-0 translate-y-3', 
    'opacity-0 translate-y-3', 'opacity-0 translate-y-3'
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

  // 5 Parameters - Glass morphism dark mode with color accents
  const parameters = [
    {
      title: 'TEMPERATURE',
      value: '44.5',
      unit: '°C',
      subtext: 'Δ +0.8°C since last update',
      confidence: 'Model confidence 94%',
      accentColor: '#0066CC',
      trend: 'up',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      )
    },
    {
      title: 'PRESSURE',
      value: '2.1',
      unit: ' bar',
      subtext: 'Δ -0.2 bar predicted',
      confidence: 'Accuracy 96%',
      accentColor: '#004C99',
      trend: 'down',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      )
    },
    {
      title: 'HUMIDITY',
      value: '58',
      unit: '%',
      subtext: 'Trending upward',
      confidence: 'Model confidence 92%',
      accentColor: '#00D9C0',
      trend: 'up',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
        </svg>
      )
    },
    {
      title: 'VIBRATION',
      value: '2.3',
      unit: ' mm/s',
      subtext: 'RMS velocity',
      confidence: 'Accuracy 89%',
      accentColor: '#1E88E5',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      )
    },
    {
      title: 'ADD PARAMETER',
      value: '+',
      unit: '',
      subtext: 'Configure new sensor',
      confidence: '',
      accentColor: '#FFFFFF',
      isDashed: true,
      icon: null
    }
  ]

  return (
    <div className="flex min-h-screen" style={{ background: 'linear-gradient(180deg, #0A0E24 0%, #0D1533 100%)' }}>
      <Sidebar activeSection="dashboard" />

      <div className="flex-1 flex flex-col">
        {/* Glass Top Navigation Bar */}
        <header 
          className="h-16"
          style={{ 
            background: 'rgba(0, 76, 153, 0.65)',
            backdropFilter: 'blur(14px)',
            borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
          }}
        >
          <div className="flex items-center justify-between px-8 h-full">
            <div className="flex items-center space-x-3">
              <div 
                className="w-9 h-9 rounded-lg flex items-center justify-center"
                style={{ 
                  background: 'rgba(0, 217, 192, 0.2)',
                  border: '1px solid rgba(0, 217, 192, 0.4)'
                }}
              >
                <span className="text-[#00D9C0] font-bold text-base" style={{ fontFamily: 'Inter, sans-serif' }}>E</span>
              </div>
              <h1 
                className="text-base font-semibold"
                style={{ color: '#E0E6F0', fontFamily: 'Inter, sans-serif' }}
              >
                Predictive Maintenance | Live Overview
              </h1>
            </div>
            <div className="flex items-center space-x-6">
              <div 
                className="flex items-center text-sm font-medium"
                style={{ color: '#E0E6F0', fontFamily: 'Inter, sans-serif' }}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>{currentTime}</span>
              </div>
              <button 
                className="relative p-2 transition-all duration-300"
                style={{ color: '#E0E6F0' }}
                onMouseEnter={(e) => e.currentTarget.style.boxShadow = '0 0 6px #00D9C0'}
                onMouseLeave={(e) => e.currentTarget.style.boxShadow = 'none'}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                </svg>
                <span className="absolute top-1 right-1 w-2 h-2 rounded-full" style={{ background: '#FF6B35' }}></span>
              </button>
              <div 
                className="w-9 h-9 rounded-full flex items-center justify-center cursor-pointer hover:scale-105 transition-transform"
                style={{ 
                  background: 'rgba(0, 217, 192, 0.2)',
                  border: '2px solid rgba(0, 217, 192, 0.5)'
                }}
              >
                <span className="text-[#00D9C0] text-xs font-semibold" style={{ fontFamily: 'Inter, sans-serif' }}>JD</span>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 p-8">
          {/* Section Title */}
          <div className="mb-8">
            <h2 
              className="text-xl font-bold mb-1"
              style={{ color: '#E0E6F0', fontFamily: 'Inter, sans-serif' }}
            >
              System Overview
            </h2>
            <p 
              className="text-sm"
              style={{ color: '#93A1C6', fontFamily: 'Inter, sans-serif' }}
            >
              TCN Model v2.3 (Active)
            </p>
          </div>

          {/* Parameter Cards Grid - 3 top, 2 bottom */}
          <div className="max-w-7xl">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {parameters.map((param, index) => (
                <div
                  key={index}
                  className={`rounded-2xl p-7 cursor-pointer transition-all duration-300 ease-in-out ${
                    param.isDashed
                      ? 'border-2 border-dashed'
                      : 'hover:-translate-y-1'
                  } ${cardAnimationClasses[index]}`}
                  style={{
                    background: param.isDashed 
                      ? 'rgba(255, 255, 255, 0.05)' 
                      : 'rgba(255, 255, 255, 0.07)',
                    backdropFilter: 'blur(14px)',
                    border: param.isDashed 
                      ? '2px dashed rgba(255, 255, 255, 0.3)' 
                      : '1px solid rgba(255, 255, 255, 0.1)',
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.25)',
                    borderRadius: '16px'
                  }}
                  onMouseEnter={(e) => {
                    if (!param.isDashed) {
                      e.currentTarget.style.boxShadow = `0 6px 20px ${param.accentColor}40`
                      e.currentTarget.style.outline = `1px solid ${param.accentColor}80`
                      e.currentTarget.style.outlineOffset = '2px'
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!param.isDashed) {
                      e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.25)'
                      e.currentTarget.style.outline = 'none'
                    }
                  }}
                >
                  {param.isDashed ? (
                    <div className="flex flex-col items-center justify-center h-full min-h-[200px]">
                      <div 
                        className="w-14 h-14 rounded-full flex items-center justify-center mb-4 transition-all duration-300 hover:scale-110"
                        style={{ background: 'rgba(255, 255, 255, 0.1)' }}
                      >
                        <span className="text-3xl font-light" style={{ color: '#E0E6F0' }}>+</span>
                      </div>
                      <h3 
                        className="text-sm font-semibold mb-2"
                        style={{ color: '#B0C4DE', fontFamily: 'Inter, sans-serif', fontWeight: 600 }}
                      >
                        {param.title}
                      </h3>
                      <p className="text-xs" style={{ color: '#93A1C6', fontFamily: 'Inter, sans-serif' }}>
                        {param.subtext}
                      </p>
                    </div>
                  ) : (
                    <div>
                      {/* Top row - icon + title + trend */}
                      <div className="flex items-center justify-between mb-5">
                        <div className="flex items-center space-x-3">
                          <div 
                            className="w-10 h-10 rounded-lg flex items-center justify-center"
                            style={{ 
                              background: `${param.accentColor}20`,
                              color: param.accentColor
                            }}
                          >
                            {param.icon}
                          </div>
                          <h3 
                            className="text-sm font-semibold uppercase tracking-wide"
                            style={{ color: '#B0C4DE', fontFamily: 'Inter, sans-serif', fontWeight: 600 }}
                          >
                            {param.title}
                          </h3>
                        </div>
                        {param.trend && (
                          <svg 
                            xmlns="http://www.w3.org/2000/svg" 
                            className="h-5 w-5" 
                            fill="none" 
                            viewBox="0 0 24 24" 
                            stroke="currentColor"
                            style={{ 
                              color: param.trend === 'up' ? '#00D9C0' : '#FF6B35',
                              transform: param.trend === 'down' ? 'rotate(180deg)' : 'none'
                            }}
                          >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                          </svg>
                        )}
                      </div>
                      
                      {/* Large metric value */}
                      <p 
                        className="mb-4"
                        style={{ 
                          color: param.accentColor === '#00D9C0' ? param.accentColor : '#FFFFFF',
                          fontFamily: 'Poppins, sans-serif', 
                          fontSize: '42px',
                          fontWeight: 700,
                          lineHeight: 1
                        }}
                      >
                        {param.value}<span style={{ fontSize: '28px' }}>{param.unit}</span>
                      </p>
                      
                      {/* Mini sparkline placeholder */}
                      <div 
                        className="w-full h-8 mb-3 rounded"
                        style={{
                          background: `linear-gradient(90deg, ${param.accentColor}15 0%, ${param.accentColor}05 100%)`,
                          position: 'relative',
                          overflow: 'hidden'
                        }}
                      >
                        <svg className="w-full h-full" viewBox="0 0 100 30" preserveAspectRatio="none">
                          <path 
                            d="M0,15 Q25,8 50,12 T100,10" 
                            fill="none" 
                            stroke={param.accentColor}
                            strokeWidth="1.5"
                            opacity="0.6"
                          />
                        </svg>
                      </div>
                      
                      {/* Subtext */}
                      <p 
                        className="text-xs mb-1"
                        style={{ color: '#93A1C6', fontFamily: 'Inter, sans-serif' }}
                      >
                        {param.subtext}
                      </p>
                      
                      {/* Confidence */}
                      {param.confidence && (
                        <p 
                          className="text-xs"
                          style={{ color: '#93A1C6', fontFamily: 'Inter, sans-serif' }}
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
