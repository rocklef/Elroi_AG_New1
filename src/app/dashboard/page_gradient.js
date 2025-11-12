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
    'opacity-0 scale-95', 'opacity-0 scale-95', 'opacity-0 scale-95', 
    'opacity-0 scale-95', 'opacity-0 scale-95'
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
      setCurrentTime(now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }))
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

    // Staggered card animations with 0.1s delay
    const timers = [100, 200, 300, 400, 500].map((delay, index) => {
      return setTimeout(() => {
        setCardAnimationClasses(prev => {
          const newClasses = [...prev]
          newClasses[index] = 'opacity-100 scale-100'
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

  // 5 Parameters - Temperature, Pressure, Humidity, Vibration, Add Parameter
  const parameters = [
    {
      title: 'Temperature',
      value: '44.5',
      unit: '°C',
      subtext: 'Δ +0.8°C since last update',
      confidence: 'Model confidence 94%',
      baseColor: '#0071CE',
      lightColor: '#E3F2FD',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 00200-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      )
    },
    {
      title: 'Pressure',
      value: '2.1',
      unit: ' bar',
      subtext: 'Δ -0.2 bar predicted',
      confidence: 'Accuracy 96%',
      baseColor: '#1A73E8',
      lightColor: '#E8F0FE',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      )
    },
    {
      title: 'Humidity',
      value: '58',
      unit: '%',
      subtext: '↑ trending upward',
      confidence: 'Model confidence 92%',
      baseColor: '#00ACC1',
      lightColor: '#E0F7FA',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
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
      baseColor: '#5C6BC0',
      lightColor: '#E8EAF6',
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
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
      baseColor: 'transparent',
      lightColor: 'white',
      isDashed: true,
      icon: null
    }
  ]

  return (
    <div className="flex min-h-screen" style={{ background: 'linear-gradient(180deg, #F8FAFF 0%, #E9EEF9 100%)' }}>
      <Sidebar activeSection="dashboard" />

      <div className="flex-1 flex flex-col">
        {/* Floating Top Navigation Bar */}
        <header 
          className="mx-6 mt-4 mb-2 rounded-2xl transition-all duration-300"
          style={{ 
            background: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(20px)',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
            borderBottom: '2px solid #0071CE'
          }}
        >
          <div className="flex items-center justify-between px-8 py-4">
            <div>
              <h1 className="text-xl font-semibold text-[#081440]">ELROI Predictive Maintenance</h1>
              <div className="flex items-center space-x-2 text-sm text-gray-500 mt-1">
                <span className="w-2 h-2 rounded-full bg-[#0071CE] animate-pulse"></span>
                <span>Live Overview</span>
              </div>
            </div>
            <div className="flex items-center space-x-6">
              <div className="flex items-center text-sm text-gray-600 space-x-4">
                <div className="flex items-center space-x-2">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>{currentTime}</span>
                </div>
              </div>
              <button className="relative p-2 text-gray-600 hover:text-[#0071CE] transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                </svg>
                <span className="absolute top-1 right-1 w-2 h-2 bg-[#E53935] rounded-full"></span>
              </button>
              <button className="p-2 text-gray-600 hover:text-[#0071CE] transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              </button>
              <div 
                className="w-10 h-10 rounded-full flex items-center justify-center cursor-pointer hover:scale-110 transition-transform"
                style={{ 
                  background: '#0071CE',
                  boxShadow: '0 4px 12px rgba(0, 113, 206, 0.4)'
                }}
              >
                <span className="text-white text-sm font-semibold">JD</span>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 px-8 py-6">
          {/* Parameters Grid - 2x3 layout */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {parameters.map((param, index) => (
              <div
                key={index}
                className={`rounded-3xl p-8 cursor-pointer transition-all duration-300 ease-out ${
                  param.isDashed
                    ? 'border-2 border-dashed border-gray-300 hover:border-[#0071CE] hover:bg-gray-50'
                    : 'hover:scale-105'
                } ${cardAnimationClasses[index]}`}
                style={{
                  background: param.isDashed 
                    ? 'white' 
                    : `linear-gradient(135deg, ${param.baseColor} 0%, ${param.lightColor} 100%)`,
                  boxShadow: param.isDashed ? 'none' : '0 8px 24px rgba(0, 0, 0, 0.1)',
                  position: 'relative',
                  overflow: 'hidden'
                }}
                onMouseEnter={(e) => {
                  if (!param.isDashed) {
                    e.currentTarget.style.boxShadow = `0 12px 32px ${param.baseColor}40`
                  }
                }}
                onMouseLeave={(e) => {
                  if (!param.isDashed) {
                    e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 0, 0, 0.1)'
                  }
                }}
              >
                {/* Subtle sparkline background */}
                {!param.isDashed && (
                  <div 
                    className="absolute inset-0 opacity-10"
                    style={{
                      backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 30'%3E%3Cpath d='M0,15 Q25,5 50,15 T100,15' stroke='white' fill='none' stroke-width='2'/%3E%3C/svg%3E")`,
                      backgroundSize: 'cover',
                      backgroundPosition: 'center'
                    }}
                  />
                )}

                {param.isDashed ? (
                  <div className="flex flex-col items-center justify-center h-full min-h-[160px]">
                    <div className="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center mb-4 transition-transform duration-300 hover:scale-110">
                      <span className="text-4xl text-gray-400 font-light">+</span>
                    </div>
                    <h3 className="text-gray-700 text-base font-semibold mb-1">{param.title}</h3>
                    <p className="text-gray-400 text-sm">{param.subtext}</p>
                  </div>
                ) : (
                  <>
                    <div className="flex items-start justify-between mb-6">
                      <div 
                        className="w-12 h-12 rounded-xl flex items-center justify-center"
                        style={{ 
                          background: 'rgba(255, 255, 255, 0.3)',
                          color: param.baseColor
                        }}
                      >
                        {param.icon}
                      </div>
                    </div>
                    <h3 
                      className="text-base font-medium mb-3"
                      style={{ color: param.baseColor }}
                    >
                      {param.title}
                    </h3>
                    <p className="text-5xl font-bold mb-2" style={{ color: param.baseColor }}>
                      {param.value}<span className="text-3xl">{param.unit}</span>
                    </p>
                    <p className="text-sm opacity-80 mb-1" style={{ color: param.baseColor }}>
                      {param.subtext}
                    </p>
                    <p className="text-xs opacity-60" style={{ color: param.baseColor }}>
                      {param.confidence}
                    </p>
                  </>
                )}
              </div>
            ))}
          </div>
        </main>
      </div>
    </div>
  )
}
