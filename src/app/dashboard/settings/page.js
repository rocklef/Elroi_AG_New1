'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { supabase } from '@/lib/supabase'
import Sidebar from '@/components/Sidebar'

export default function SettingsPage() {
  const [user, setUser] = useState(null)
  const router = useRouter()
  const [animationClass, setAnimationClass] = useState('opacity-0 translate-y-4')

  useEffect(() => {
    // Trigger entrance animation
    setTimeout(() => {
      setAnimationClass('opacity-100 translate-y-0')
    }, 100)
  }, [])

  const [settings, setSettings] = useState({
    notifications: true,
    emailAlerts: true,
    darkMode: false,
    autoRefresh: true,
    thresholdTemp: 35,
    thresholdPressure: 2.5,
    thresholdHumidity: 65,
    thresholdVibration: 1.2
  })

  // Load settings from localStorage on mount
  useEffect(() => {
    const savedSettings = localStorage.getItem('systemSettings')
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings))
    }
  }, [])

  const handleSave = () => {
    // Save to localStorage
    localStorage.setItem('systemSettings', JSON.stringify(settings))
    
    // Trigger storage event for other tabs/components
    window.dispatchEvent(new Event('storage'))
    
    alert('Settings saved successfully! All thresholds updated.')
  }

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-cyan-50">
      {/* Left Sidebar */}
      <Sidebar activeSection="settings" />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Main Content */}
        <main className={`flex-1 p-8 transition-all duration-700 ease-in-out ${animationClass}`}>
          {/* Page Title */}
          <div className="mb-8 animate-fade-in-up">
            <h1 className="text-2xl font-bold text-[#1A1F36]">System Configuration</h1>
            <p className="text-gray-600 text-sm">Manage your preferences and system thresholds</p>
          </div>

          <div className="border-b border-gray-200 mb-8"></div>

          {/* Settings Cards */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Preferences Card */}
            <div className="bg-gradient-to-br from-blue-100 via-blue-50 to-cyan-50 rounded-2xl border-2 border-blue-200 shadow-xl p-6 transition-all duration-500 hover:shadow-2xl hover:scale-105 animate-fade-in-up delay-100">
              <div className="flex items-center mb-6">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-600 flex items-center justify-center mr-3 shadow-lg shadow-blue-500/40">
                  <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <h2 className="text-[#1A1F36] text-xl font-black">Preferences</h2>
              </div>
              
              <div className="space-y-6">
                <div className="flex items-center justify-between transition-all duration-200 hover:bg-white/60 p-4 rounded-xl">
                  <div>
                    <h3 className="text-[#1A1F36] font-bold">Notifications</h3>
                    <p className="text-gray-600 text-sm">Receive system alerts</p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input 
                      type="checkbox" 
                      className="sr-only peer" 
                      checked={settings.notifications}
                      onChange={(e) => setSettings({...settings, notifications: e.target.checked})}
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[#1E73BE]"></div>
                  </label>
                </div>
                
                <div className="flex items-center justify-between transition-all duration-200 hover:bg-white/60 p-4 rounded-xl">
                  <div>
                    <h3 className="text-[#1A1F36] font-bold">Email Alerts</h3>
                    <p className="text-gray-600 text-sm">Send alerts to your email</p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input 
                      type="checkbox" 
                      className="sr-only peer" 
                      checked={settings.emailAlerts}
                      onChange={(e) => setSettings({...settings, emailAlerts: e.target.checked})}
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[#1E73BE]"></div>
                  </label>
                </div>
                
                <div className="flex items-center justify-between transition-all duration-200 hover:bg-white/60 p-4 rounded-xl">
                  <div>
                    <h3 className="text-[#1A1F36] font-bold">Auto Refresh</h3>
                    <p className="text-gray-600 text-sm">Automatically refresh data</p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input 
                      type="checkbox" 
                      className="sr-only peer" 
                      checked={settings.autoRefresh}
                      onChange={(e) => setSettings({...settings, autoRefresh: e.target.checked})}
                    />
                    <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[#1E73BE]"></div>
                  </label>
                </div>
              </div>
            </div>

            {/* Thresholds Card */}
            <div className="bg-gradient-to-br from-purple-100 via-purple-50 to-pink-50 rounded-2xl border-2 border-purple-200 shadow-xl p-6 transition-all duration-500 hover:shadow-2xl hover:scale-105 animate-fade-in-up delay-200">
              <div className="flex items-center mb-6">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-600 to-pink-600 flex items-center justify-center mr-3 shadow-lg shadow-purple-500/40">
                  <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <h2 className="text-[#1A1F36] text-xl font-black">Sensor Thresholds</h2>
              </div>
              
              <div className="space-y-6">
                <div className="transition-all duration-200 hover:bg-white/60 p-4 rounded-xl">
                  <label className="block text-[#1A1F36] text-sm font-bold mb-2">Temperature (°C)</label>
                  <input 
                    type="number" 
                    value={settings.thresholdTemp}
                    onChange={(e) => setSettings({...settings, thresholdTemp: parseFloat(e.target.value)})}
                    className="w-full px-4 py-2 border border-[#D3D9E3] rounded-lg focus:ring-2 focus:ring-[#1E73BE] focus:border-[#1E73BE] bg-white text-gray-900 font-semibold transition-all duration-200 hover:shadow-md"
                  />
                </div>
                
                <div className="transition-all duration-200 hover:bg-white/60 p-4 rounded-xl">
                  <label className="block text-[#1A1F36] text-sm font-bold mb-2">Pressure (bar)</label>
                  <input 
                    type="number" 
                    step="0.1"
                    value={settings.thresholdPressure}
                    onChange={(e) => setSettings({...settings, thresholdPressure: parseFloat(e.target.value)})}
                    className="w-full px-4 py-2 border border-[#D3D9E3] rounded-lg focus:ring-2 focus:ring-[#1E73BE] focus:border-[#1E73BE] bg-white text-gray-900 font-semibold transition-all duration-200 hover:shadow-md"
                  />
                </div>
                
                <div className="transition-all duration-200 hover:bg-white/60 p-4 rounded-xl">
                  <label className="block text-[#1A1F36] text-sm font-bold mb-2">Humidity (%)</label>
                  <input 
                    type="number" 
                    value={settings.thresholdHumidity}
                    onChange={(e) => setSettings({...settings, thresholdHumidity: parseFloat(e.target.value)})}
                    className="w-full px-4 py-2 border border-[#D3D9E3] rounded-lg focus:ring-2 focus:ring-[#1E73BE] focus:border-[#1E73BE] bg-white text-gray-900 font-semibold transition-all duration-200 hover:shadow-md"
                  />
                </div>
                
                <div className="transition-all duration-200 hover:bg-white/60 p-4 rounded-xl">
                  <label className="block text-[#1A1F36] text-sm font-bold mb-2">Vibration (mm/s)</label>
                  <input 
                    type="number" 
                    step="0.1"
                    value={settings.thresholdVibration}
                    onChange={(e) => setSettings({...settings, thresholdVibration: parseFloat(e.target.value)})}
                    className="w-full px-4 py-2 border border-[#D3D9E3] rounded-lg focus:ring-2 focus:ring-[#1E73BE] focus:border-[#1E73BE] bg-white text-gray-900 font-semibold transition-all duration-200 hover:shadow-md"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Save Button */}
          <div className="mt-8 flex justify-end animate-fade-in-up delay-300">
            <button 
              onClick={handleSave}
              className="px-8 py-3 bg-gradient-to-r from-blue-600 via-blue-700 to-cyan-700 text-white font-bold rounded-xl hover:from-blue-700 hover:to-cyan-800 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transform hover:scale-105 hover:shadow-xl shadow-lg shadow-blue-500/40"
            >
              💾 Save Settings
            </button>
          </div>
        </main>
      </div>
    </div>
  )
}