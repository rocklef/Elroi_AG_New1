'use client'

import { useState, useEffect } from 'react'
import Sidebar from '@/components/Sidebar'
import TopNav from '@/components/TopNav'
import { useToast } from '@/components/ui/ToastContext'

export default function AlertsPage() {
  const { addToast } = useToast()
  const [activeTab, setActiveTab] = useState('history')
  const [alertHistory, setAlertHistory] = useState([])
  const [alertConfig, setAlertConfig] = useState({
    leadTimeMinutes: 15,
    customMessage: '',
    thresholdTemp: 31.7,
    currentTemp: 35.2,
    recipients: [
      {
        name: 'Admin',
        position: 'System Administrator',
        email: 'tb2138@srmist.edu.in'
      }
    ]
  })
  const [editingRecipient, setEditingRecipient] = useState(null)
  const [newRecipient, setNewRecipient] = useState({
    name: '',
    position: '',
    email: ''
  })

  // Load data from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('alertHistory')
    if (savedHistory) {
      try {
        setAlertHistory(JSON.parse(savedHistory))
      } catch (e) {
        console.error('Error loading alert history:', e)
      }
    }

    const savedConfig = localStorage.getItem('alertConfig')
    if (savedConfig) {
      try {
        const config = JSON.parse(savedConfig)
        if (config.emails && !config.recipients) {
          config.recipients = config.emails.map(email => ({
            name: email.split('@')[0],
            position: '',
            email: email
          }))
          delete config.emails
        }
        setAlertConfig(config)
      } catch (e) {
        console.error('Error loading alert config:', e)
      }
    } else {
      // No saved config, save the default recipient
      const defaultConfig = {
        leadTimeMinutes: 15,
        customMessage: '',
        thresholdTemp: 31.7,
        currentTemp: 35.2,
        recipients: [
          {
            name: 'Admin',
            position: 'System Administrator',
            email: 'tb2138@srmist.edu.in'
          }
        ]
      }
      localStorage.setItem('alertConfig', JSON.stringify(defaultConfig))
    }
  }, [])

  // Save config to localStorage
  const saveConfig = () => {
    localStorage.setItem('alertConfig', JSON.stringify(alertConfig))
    addToast('Alert configuration saved successfully!', 'success')
  }

  // Add new recipient
  const handleAddRecipient = () => {
    if (!newRecipient.name.trim() || !newRecipient.email.trim()) {
      addToast('Please fill in Name and Email fields', 'warning')
      return
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(newRecipient.email)) {
      addToast('Please enter a valid email format', 'warning')
      return
    }
    if (alertConfig.recipients.some(r => r.email === newRecipient.email)) {
      addToast('This email already exists', 'warning')
      return
    }
    setAlertConfig(prev => ({
      ...prev,
      recipients: [...prev.recipients, { ...newRecipient }]
    }))
    setNewRecipient({ name: '', position: '', email: '' })
  }

  // Update recipient
  const handleUpdateRecipient = (index, updatedRecipient) => {
    const updated = [...alertConfig.recipients]
    updated[index] = updatedRecipient
    setAlertConfig(prev => ({
      ...prev,
      recipients: updated
    }))
  }

  // Delete recipient
  const handleDeleteRecipient = (index) => {
    if (window.confirm('Are you sure you want to remove this recipient?')) {
      setAlertConfig(prev => ({
        ...prev,
        recipients: prev.recipients.filter((_, i) => i !== index)
      }))
    }
  }

  // Resend alert to specific recipient
  const handleResendAlert = async (recipient) => {
    try {
      const response = await fetch('/api/send-alert', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          emails: [recipient.email],
          currentTemp: alertConfig.currentTemp,
          threshold: alertConfig.thresholdTemp,
          etaMinutes: 12,
          customMessage: alertConfig.customMessage,
          recipientNames: [recipient.name],
          isDanger: alertConfig.currentTemp <= alertConfig.thresholdTemp
        })
      })

      const data = await response.json()
      if (data.success) {
        const alertHistory = JSON.parse(localStorage.getItem('alertHistory') || '[]')
        alertHistory.push({
          email: recipient.email,
          name: recipient.name,
          currentTemp: alertConfig.currentTemp,
          threshold: alertConfig.thresholdTemp,
          etaMinutes: 12,
          customMessage: alertConfig.customMessage,
          timestamp: new Date().toLocaleString()
        })
        localStorage.setItem('alertHistory', JSON.stringify(alertHistory))
        setAlertHistory(alertHistory)
        addToast('Alert sent', 'success', 'send')
      }
    } catch (error) {
      console.error('Error resending alert:', error)
      addToast('❌ Failed to resend alert', 'error')
    }
  }

  // Clear alert history
  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear all alert history?')) {
      setAlertHistory([])
      localStorage.removeItem('alertHistory')
    }
  }

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-cyan-50">
      <Sidebar activeSection="alerts" />

      <div className="flex-1 flex flex-col overflow-hidden">
        <TopNav />

        <div className="flex-1 overflow-auto">
          <div className="max-w-7xl mx-auto px-8 py-10">
            {/* Header with gradient background */}
            <div className="mb-10 bg-gradient-to-r from-blue-600 via-cyan-600 to-teal-600 rounded-2xl p-8 text-white shadow-lg">
              <h1 className="text-4xl font-bold mb-2">🚨 Alert Management</h1>
              <p className="text-blue-100 text-lg">Configure temperature alerts, manage recipients, and track alert history</p>
            </div>

            {/* Tabs with enhanced styling */}
            <div className="flex gap-2 mb-8 bg-white rounded-xl p-1 shadow-md border-2 border-blue-200 w-fit">
              <button
                onClick={() => setActiveTab('history')}
                className={`cursor-pointer px-8 py-3 font-bold text-base rounded-lg transition-all duration-150 active:scale-95 ${activeTab === 'history'
                  ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-lg scale-105'
                  : 'text-gray-700 hover:text-gray-900 hover:bg-gray-100'
                  }`}
              >
                📋 Alert History
              </button>
              <button
                onClick={() => setActiveTab('config')}
                className={`cursor-pointer px-8 py-3 font-bold text-base rounded-lg transition-all duration-150 active:scale-95 ${activeTab === 'config'
                  ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-lg scale-105'
                  : 'text-gray-700 hover:text-gray-900 hover:bg-gray-100'
                  }`}
              >
                ⚙️ Configuration
              </button>
            </div>

            {/* HISTORY TAB */}
            {activeTab === 'history' && (
              <div className="space-y-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-bold text-gray-900">📊 Sent Alerts Log</h2>
                  {alertHistory.length > 0 && (
                    <button
                      onClick={handleClearHistory}
                      className="px-6 py-2.5 bg-red-500 hover:bg-red-600 text-white font-semibold rounded-lg transition-all shadow-md hover:shadow-lg transform hover:scale-105"
                    >
                      🗑️ Clear History
                    </button>
                  )}
                </div>

                {alertHistory.length === 0 ? (
                  <div className="text-center py-16 bg-white rounded-2xl border-2 border-dashed border-gray-300 shadow-sm">
                    <div className="text-6xl mb-4 opacity-50">📭</div>
                    <p className="text-gray-600 text-lg font-semibold">No alerts sent yet</p>
                    <p className="text-gray-500 text-sm mt-2">Configure recipients to start receiving alerts</p>
                  </div>
                ) : (
                  <div className="bg-white rounded-2xl border-2 border-blue-300 overflow-hidden shadow-lg">
                    <table className="w-full">
                      <thead>
                        <tr className="bg-gradient-to-r from-blue-600 to-cyan-600 text-white">
                          <th className="px-6 py-4 text-left text-sm font-bold">📧 Recipient</th>
                          <th className="px-6 py-4 text-left text-sm font-bold">🌡️ Current Temp</th>
                          <th className="px-6 py-4 text-left text-sm font-bold">⚠️ Threshold</th>
                          <th className="px-6 py-4 text-left text-sm font-bold">⏱️ ETA</th>
                          <th className="px-6 py-4 text-left text-sm font-bold">📅 Sent At</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-blue-200">
                        {alertHistory.map((alert, idx) => (
                          <tr key={idx} className="hover:bg-blue-50 transition-colors">
                            <td className="px-6 py-4 text-sm font-semibold text-gray-900">{alert.email}</td>
                            <td className="px-6 py-4 text-sm font-bold text-red-600 text-lg">{alert.currentTemp?.toFixed(2)}°C</td>
                            <td className="px-6 py-4 text-sm font-semibold text-gray-800">{alert.threshold?.toFixed(2)}°C</td>
                            <td className="px-6 py-4 text-sm font-bold text-orange-600">{alert.etaMinutes} min</td>
                            <td className="px-6 py-4 text-sm text-gray-700">{alert.timestamp}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {/* CONFIGURATION TAB */}
            {activeTab === 'config' && (
              <div className="space-y-8">

                {/* Recipients Card */}
                <div className="bg-gradient-to-br from-white to-teal-50 p-8 rounded-2xl border-2 border-teal-400 shadow-lg">
                  <div className="flex items-center gap-3 mb-6">
                    <span className="text-3xl">👥</span>
                    <h3 className="text-lg font-bold text-gray-900">Alert Recipients</h3>
                  </div>

                  {/* Add New Recipient Form */}
                  <div className="mb-8 p-6 bg-gradient-to-br from-teal-50 to-cyan-50 rounded-xl border-2 border-teal-500 shadow-md">
                    <h4 className="font-bold text-gray-900 mb-5 text-base flex items-center gap-2">
                      <span className="text-xl">➕</span> Add New Recipient
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                      <div>
                        <label className="block text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">👤 Name</label>
                        <input
                          type="text"
                          value={newRecipient.name}
                          onChange={(e) => setNewRecipient(prev => ({ ...prev, name: e.target.value }))}
                          placeholder="Full Name"
                          className="w-full px-4 py-3 border-2 border-teal-500 rounded-lg text-sm text-gray-900 bg-white font-medium focus:ring-2 focus:ring-teal-400 focus:border-teal-500 shadow-sm placeholder-gray-600"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">💼 Position</label>
                        <input
                          type="text"
                          value={newRecipient.position}
                          onChange={(e) => setNewRecipient(prev => ({ ...prev, position: e.target.value }))}
                          placeholder="Job Title"
                          className="w-full px-4 py-3 border-2 border-teal-500 rounded-lg text-sm text-gray-900 bg-white font-medium focus:ring-2 focus:ring-teal-400 focus:border-teal-500 shadow-sm placeholder-gray-600"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-bold text-gray-900 mb-2 uppercase tracking-wide">📧 Email ID</label>
                        <input
                          type="email"
                          value={newRecipient.email}
                          onChange={(e) => setNewRecipient(prev => ({ ...prev, email: e.target.value }))}
                          placeholder="email@example.com"
                          className="w-full px-4 py-3 border-2 border-teal-500 rounded-lg text-sm text-gray-900 bg-white font-medium focus:ring-2 focus:ring-teal-400 focus:border-teal-500 shadow-sm placeholder-gray-600"
                        />
                      </div>
                      <div className="flex items-end">
                        <button
                          onClick={handleAddRecipient}
                          className="cursor-pointer w-full px-4 py-3 bg-gradient-to-r from-teal-500 to-teal-600 hover:from-teal-600 hover:to-teal-700 text-white font-bold rounded-lg transition-all duration-150 shadow-md hover:shadow-lg active:scale-95 transform hover:scale-105"
                        >
                          ➕ Add
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Recipients List Table */}
                  {alertConfig.recipients.length === 0 ? (
                    <div className="text-center py-10 text-gray-600 bg-gray-50 rounded-lg border-2 border-dashed border-gray-400">
                      <p className="text-lg font-semibold">No recipients added yet</p>
                      <p className="text-sm text-gray-500 mt-1">Add your first recipient above to start receiving alerts</p>
                    </div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="bg-gradient-to-r from-teal-600 to-teal-700 text-white">
                            <th className="px-5 py-4 text-left font-bold">👤 Name</th>
                            <th className="px-5 py-4 text-left font-bold">💼 Position</th>
                            <th className="px-5 py-4 text-left font-bold">📧 Email ID</th>
                            <th className="px-5 py-4 text-center font-bold">⚡ Actions</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-teal-300">
                          {alertConfig.recipients.map((recipient, idx) => (
                            <tr key={idx} className="hover:bg-teal-50 transition-colors">
                              <td className="px-5 py-4">
                                {editingRecipient === idx ? (
                                  <input
                                    type="text"
                                    value={recipient.name}
                                    onChange={(e) => handleUpdateRecipient(idx, { ...recipient, name: e.target.value })}
                                    className="w-full px-3 py-2 border-2 border-blue-500 rounded-lg text-gray-900 font-semibold focus:ring-2 focus:ring-blue-400 focus:border-blue-500"
                                  />
                                ) : (
                                  <span className="text-gray-900 font-bold">{recipient.name}</span>
                                )}
                              </td>
                              <td className="px-5 py-4">
                                {editingRecipient === idx ? (
                                  <input
                                    type="text"
                                    value={recipient.position}
                                    onChange={(e) => handleUpdateRecipient(idx, { ...recipient, position: e.target.value })}
                                    className="w-full px-3 py-2 border-2 border-blue-500 rounded-lg text-gray-900 font-semibold focus:ring-2 focus:ring-blue-400 focus:border-blue-500"
                                  />
                                ) : (
                                  <span className="text-gray-800 font-semibold">{recipient.position || '—'}</span>
                                )}
                              </td>
                              <td className="px-5 py-4">
                                {editingRecipient === idx ? (
                                  <input
                                    type="email"
                                    value={recipient.email}
                                    onChange={(e) => handleUpdateRecipient(idx, { ...recipient, email: e.target.value })}
                                    className="w-full px-3 py-2 border-2 border-blue-500 rounded-lg text-gray-900 font-semibold focus:ring-2 focus:ring-blue-400 focus:border-blue-500"
                                  />
                                ) : (
                                  <span className="text-gray-800 font-semibold">{recipient.email}</span>
                                )}
                              </td>
                              <td className="px-5 py-4">
                                <div className="flex items-center justify-center gap-2 flex-wrap">
                                  {editingRecipient === idx ? (
                                    <>
                                      <button
                                        onClick={() => setEditingRecipient(null)}
                                        className="cursor-pointer px-3 py-2 bg-green-500 hover:bg-green-600 text-white text-xs font-bold rounded-lg transition-all duration-150 shadow-md hover:shadow-lg active:scale-95"
                                      >
                                        ✓ Save
                                      </button>
                                      <button
                                        onClick={() => setEditingRecipient(null)}
                                        className="cursor-pointer px-3 py-2 bg-gray-400 hover:bg-gray-500 text-white text-xs font-bold rounded-lg transition-all duration-150 shadow-md hover:shadow-lg active:scale-95"
                                      >
                                        ✕ Cancel
                                      </button>
                                    </>
                                  ) : (
                                    <>
                                      <button
                                        onClick={() => setEditingRecipient(idx)}
                                        className="cursor-pointer px-3 py-2 bg-blue-300 hover:bg-blue-400 text-blue-900 text-xs font-bold rounded-lg transition-all duration-150 shadow-md hover:shadow-lg active:scale-95 transform hover:scale-105"
                                      >
                                        ✏️ Edit
                                      </button>
                                      <button
                                        onClick={() => handleResendAlert(recipient)}
                                        className="cursor-pointer px-3 py-2 bg-orange-300 hover:bg-orange-400 text-orange-900 text-xs font-bold rounded-lg transition-all duration-150 shadow-md hover:shadow-lg active:scale-95 transform hover:scale-105"
                                      >
                                        📤 Resend
                                      </button>
                                      <button
                                        onClick={() => handleDeleteRecipient(idx)}
                                        className="cursor-pointer px-3 py-2 bg-red-300 hover:bg-red-400 text-red-900 text-xs font-bold rounded-lg transition-all duration-150 shadow-md hover:shadow-lg active:scale-95 transform hover:scale-105"
                                      >
                                        🗑️ Delete
                                      </button>
                                    </>
                                  )}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>

                {/* Save Button */}
                <div className="flex gap-4 justify-end">
                  <button
                    onClick={saveConfig}
                    className="cursor-pointer px-8 py-3.5 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white font-bold text-lg rounded-lg transition-all duration-150 shadow-lg hover:shadow-xl active:scale-95 transform hover:scale-105"
                  >
                    💾 Save All Changes
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
