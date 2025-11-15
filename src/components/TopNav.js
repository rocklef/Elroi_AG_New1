'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { supabase } from '@/lib/supabase'
import Image from 'next/image'

export default function TopNav({ title = "Dashboard" }) {
  const [user, setUser] = useState(null)
  const [currentTime, setCurrentTime] = useState('')
  const [notifications] = useState(2) // Mock notification count
  const [mounted, setMounted] = useState(false)
  const router = useRouter()

  useEffect(() => {
    setMounted(true)
    
    const getUser = async () => {
      const { data: { session } } = await supabase.auth.getSession()
      if (session?.user) {
        setUser(session.user)
      }
    }
    getUser()

    // Update time every second with full date and time
    const updateTime = () => {
      const now = new Date()
      const dateOptions = { 
        weekday: 'short', 
        year: 'numeric',
        month: 'short', 
        day: 'numeric'
      }
      const timeOptions = {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true
      }
      const dateStr = now.toLocaleDateString('en-US', dateOptions)
      const timeStr = now.toLocaleTimeString('en-US', timeOptions)
      setCurrentTime(`${dateStr} • ${timeStr}`)
    }
    updateTime()
    const timer = setInterval(updateTime, 1000)

    return () => clearInterval(timer)
  }, [])

  // Get user initials from email
  const getInitials = (email) => {
    if (!email) return 'U'
    const name = email.split('@')[0]
    const parts = name.split(/[._-]/)
    if (parts.length >= 2) {
      return (parts[0][0] + parts[1][0]).toUpperCase()
    }
    return name.slice(0, 2).toUpperCase()
  }

  return (
    <div className="bg-white border-b border-gray-200 shadow-sm">
      <div className="flex items-center justify-between px-8 py-4">
        {/* Left: Title */}
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{title}</h1>
        </div>

        {/* Right: Notifications, Time */}
        <div className="flex items-center space-x-6">
          {/* Notification Bell */}
          <div className="relative cursor-pointer" onClick={() => router.push('/dashboard/alerts')}>
            <div className="w-10 h-10 rounded-lg bg-gray-50 hover:bg-gray-100 flex items-center justify-center transition-all">
              <svg className="h-6 w-6 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
              </svg>
              {notifications > 0 && (
                <span className="absolute -top-1 -right-1 w-5 h-5 bg-gradient-to-br from-red-500 to-pink-500 text-white text-xs font-bold rounded-full flex items-center justify-center shadow-lg">
                  {notifications}
                </span>
              )}
            </div>
          </div>

          {/* Real-time Date & Time - Right Corner */}
          <div className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg border border-gray-200">
            <svg className="h-5 w-5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-sm font-medium text-gray-700 whitespace-nowrap">
              {mounted ? currentTime : 'Loading...'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
