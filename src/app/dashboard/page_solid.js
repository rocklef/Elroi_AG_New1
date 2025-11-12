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
    'opacity-0 translate-y-4', 'opacity-0 translate-y-4', 'opacity-0 translate-y-4', 
    'opacity-0 translate-y-4', 'opacity-0 translate-y-4'
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

    // Staggered card animations
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

  // 5 Parameters - solid colors, aerospace industrial aesthetic
  const parameters = [
    {
      title: 'Temperature',
      value: '44.5',
      unit: '°C',
      subtext: 'Δ +0.8°C since last update',
      baseColor: '#0071CE',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      )
    },
    {
      title: 'Pressure',
      value: '2.1',
      unit: ' bar',
      subtext: 'Δ -0.2 predicted',
      baseColor: '#1A73E8',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      )
    },
    {
      title: 'Humidity',
      value: '58',
      unit: '%',
      subtext: 'trending ↑',
      baseColor: '#007C89',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
        </svg>
      )
    },
    {
      title: 'Vibration',
      value: '2.3',
      unit: ' mm/s',
      subtext: 'RMS velocity',
      baseColor: '#304FFE',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      )
    },
    {
      title: 'Add Parameter',
      value: '+',
      unit: '',
      subtext: 'Configure new sensor',
      baseColor: '#B0BEC5',
      isDashed: true,
      icon: null
    }
  ]

  return (
    <div className="flex min-h-screen" style={{ background: '#FFFFFF' }}>
      <Sidebar activeSection="dashboard" />

      <div className="flex-1 flex flex-col">
        {/* Solid Deep Blue Top Bar */}
        <header 
          className="h-16"
          style={{ 
            background: '#081440',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
          }}
        >
          <div className="flex items-center justify-between px-8 h-full">
            <h1 
              className="text-white text-lg font-semibold"
              style={{ fontFamily: 'Poppins, sans-serif' }}
            >
              ELROI Predictive Maintenance | Live Overview
            </h1>
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

        {/* Slim white separator */}
        <div style={{ height: '1px', background: '#E0E0E0' }}></div>

        {/* Main Content Area - Clean White Background */}
        <main className="flex-1 p-8" style={{ background: '#FFFFFF' }}>
          {/* Centered Content Container ~85% width */}
          <div className="max-w-7xl mx-auto">
            {/* 2x3 Grid - Equal width cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-7">
              {parameters.map((param, index) => (
                <div
                  key={index}
                  className={`rounded-2xl p-7 cursor-pointer transition-all duration-200 ${
                    param.isDashed
                      ? 'border-2 border-dashed hover:border-[#0071CE]'
                      : 'hover:-translate-y-1'
                  } ${cardAnimationClasses[index]}`}
                  style={{
                    background: param.isDashed ? 'white' : param.baseColor,
                    boxShadow: param.isDashed 
                      ? '0 4px 12px rgba(0, 0, 0, 0.06)' 
                      : '0 6px 20px rgba(0, 0, 0, 0.08)',
                    borderColor: param.isDashed ? '#B0BEC5' : 'transparent',
                    borderRadius: '16px'
                  }}
                  onMouseEnter={(e) => {
                    if (!param.isDashed) {
                      e.currentTarget.style.boxShadow = `0 8px 24px ${param.baseColor}60`
                      e.currentTarget.style.outline = `2px solid ${param.baseColor}`
                      e.currentTarget.style.outlineOffset = '2px'
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!param.isDashed) {
                      e.currentTarget.style.boxShadow = '0 6px 20px rgba(0, 0, 0, 0.08)'
                      e.currentTarget.style.outline = 'none'
                    }
                  }}
                >
                  {param.isDashed ? (
                    <div className="flex flex-col items-center justify-center h-full min-h-[220px]">
                      <div 
                        className="w-14 h-14 rounded-full flex items-center justify-center mb-4 transition-all duration-200 hover:scale-110"
                        style={{ background: '#F5F5F5' }}
                      >
                        <span className="text-3xl font-light" style={{ color: '#5C6C80' }}>+</span>
                      </div>
                      <h3 
                        className="text-base font-semibold mb-2"
                        style={{ color: '#1A1A1A', fontFamily: 'Poppins, sans-serif' }}
                      >
                        {param.title}
                      </h3>
                      <p className="text-sm" style={{ color: '#5C6C80', fontFamily: 'Inter, sans-serif' }}>
                        {param.subtext}
                      </p>
                    </div>
                  ) : (
                    <div className="min-h-[220px] flex flex-col">
                      <div className="flex items-start justify-between mb-5">
                        <div 
                          className="w-12 h-12 rounded-lg flex items-center justify-center"
                          style={{ 
                            background: 'rgba(255, 255, 255, 0.2)',
                            color: 'white'
                          }}
                        >
                          {param.icon}
                        </div>
                      </div>
                      <h3 
                        className="text-sm font-semibold mb-4 uppercase tracking-wide"
                        style={{ color: 'white', fontFamily: 'Poppins, sans-serif', opacity: 0.95 }}
                      >
                        {param.title}
                      </h3>
                      <p 
                        className="text-5xl font-bold mb-3 flex-grow"
                        style={{ color: 'white', fontFamily: 'Inter, sans-serif' }}
                      >
                        {param.value}<span className="text-3xl font-semibold">{param.unit}</span>
                      </p>
                      <p 
                        className="text-sm"
                        style={{ color: 'white', fontFamily: 'Inter, sans-serif', opacity: 0.85 }}
                      >
                        {param.subtext}
                      </p>
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
