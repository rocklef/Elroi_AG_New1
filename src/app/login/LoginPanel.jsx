'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { supabase } from '@/lib/supabase'
import { GoogleRippleButton } from '@/components/ui/google-ripple-button'

export default function LoginPanel() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [isSignUp, setIsSignUp] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const router = useRouter()

  const handleLogin = async (e) => {
    e.preventDefault()
    setError('')
    setIsSubmitting(true)

    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })

    if (error) {
      setError(error.message)
      setIsSubmitting(false)
    } else {
      // Trigger transition animation before redirect
      setIsLoggedIn(true)
      setTimeout(() => {
        router.push('/dashboard')
        router.refresh()
      }, 700) // Match the duration of our fade-out animation
    }
  }

  const handleSignUp = async (e) => {
    e.preventDefault()
    setError('')
    setIsSubmitting(true)

    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        emailRedirectTo: `${window.location.origin}/dashboard`,
      },
    })

    if (error) {
      setError(error.message)
      setIsSubmitting(false)
    } else {
      // For demo purposes, we'll redirect to dashboard after signup
      // In a real app, you might want to show a confirmation message
      setIsLoggedIn(true)
      setTimeout(() => {
        router.push('/dashboard')
        router.refresh()
      }, 700) // Match the duration of our fade-out animation
    }
  }

  const handleGoogleLogin = async () => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: `${window.location.origin}/dashboard`,
      },
    })

    if (error) {
      setError(error.message)
    }
  }

  const toggleMode = (mode) => {
    setIsSignUp(mode === 'signup')
    setError('')
  }

  return (
    <div className={`bg-white flex flex-col space-y-4 transition-all duration-700 ease-in-out ${isLoggedIn ? 'opacity-0 scale-95' : 'opacity-100 scale-100'}`}>
      {/* Header */}
      <div className="animate-fade-in-up">
        <h1 className="text-[2rem] font-bold text-[#1A1F36] leading-tight">Welcome Back</h1>
        <p className="text-[1.05rem] text-gray-600 mt-2 leading-relaxed">
          {isSignUp ? 'Sign up to get started' : 'Sign in to continue to your dashboard'}
        </p>
      </div>

      {/* Toggle bar */}
      <div className="flex bg-gray-100 rounded-md overflow-hidden rounded animate-fade-in-up delay-100">
        <button
          onClick={() => toggleMode('signup')}
          className={`w-1/2 py-2 text-base font-medium cursor-pointer transition-all duration-300 transform hover:scale-105 ${isSignUp ? 'text-white bg-[#1E73BE]' : 'text-gray-600 hover:bg-gray-200'}`}
        >
          Sign up
        </button>
        <button
          onClick={() => toggleMode('login')}
          className={`w-1/2 py-2 text-base font-medium cursor-pointer transition-all duration-300 transform hover:scale-105 ${!isSignUp ? 'text-white bg-[#1E73BE]' : 'text-gray-600 hover:bg-gray-200'}`}
        >
          Log in
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-50 text-red-700 rounded-lg text-base font-medium border border-red-100 animate-fade-in-up">
          {error}
        </div>
      )}

      {/* Form fields */}
      <form onSubmit={isSignUp ? handleSignUp : handleLogin} className="flex flex-col space-y-3">
        {/* Email */}
        <div className="animate-fade-in-up delay-100">
          <label className="text-sm font-semibold text-[#1A1F36] tracking-wide">
            EMAIL ADDRESS
          </label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full border border-[#D3D9E3] rounded-md px-4 py-3 mt-1.5 bg-[#F7F9FC] focus:outline-none focus:ring-2 focus:ring-[#1E73BE] text-base transition-all duration-300 hover:shadow-md"
            placeholder="you@company.com"
            required
            disabled={isSubmitting || isLoggedIn}
          />
        </div>

        {/* Password */}
        <div className="animate-fade-in-up delay-200">
          <label className="text-sm font-semibold text-[#1A1F36] tracking-wide flex justify-between">
            <span>PASSWORD</span>
            {!isSignUp && (
              <a href="#" className="text-[#1E73BE] text-sm hover:underline font-medium">FORGOT PASSWORD?</a>
            )}
          </label>
          <div className="relative mt-1.5">
            <input
              id="password"
              type={showPassword ? "text" : "password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full border border-[#D3D9E3] rounded-md px-4 py-3 bg-[#F7F9FC] focus:outline-none focus:ring-2 focus:ring-[#1E73BE] pr-10 text-base transition-all duration-300 hover:shadow-md"
              placeholder="••••••••"
              required
              disabled={isSubmitting || isLoggedIn}
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute inset-y-0 right-0 pr-2.5 flex items-center transition-transform duration-200 hover:scale-110"
              disabled={isSubmitting || isLoggedIn}
            >
              {showPassword ? (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-500" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
                  <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-500" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M3.707 2.293a1 1 0 00-1.414 1.414l14 14a1 1 0 001.414-1.414l-1.473-1.473A10.014 10.014 0 0019.542 10C18.268 5.943 14.478 3 10 3a9.958 9.958 0 00-4.512 1.074l-1.78-1.781zm4.261 4.26l1.514 1.515a2.003 2.003 0 012.45 2.45l1.514 1.514a4 4 0 00-5.478-5.478z" clipRule="evenodd" />
                  <path d="M12.454 16.697L9.75 13.992a4 4 0 01-3.742-3.741L2.335 6.578A9.98 9.98 0 00.458 10c1.274 4.057 5.065 7 9.542 7 .847 0 1.669-.105 2.454-.303z" />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Checkbox */}
        {!isSignUp && (
          <div className="flex items-center gap-1.5 animate-fade-in-up delay-200">
            <input
              id="showPassword"
              type="checkbox"
              checked={showPassword}
              onChange={(e) => setShowPassword(e.target.checked)}
              className="w-4 h-4 accent-[#1E73BE] rounded focus:ring-[#1E73BE] border-[#D3D9E3]"
              disabled={isSubmitting || isLoggedIn}
            />
            <label htmlFor="showPassword" className="text-sm text-gray-600">
              Show password
            </label>
          </div>
        )}

        {/* Submit button */}
        <button
          type="submit"
          disabled={isSubmitting || isLoggedIn}
          className="w-full bg-white border-2 border-[#1E73BE] text-[#1E73BE] py-3 rounded-md font-semibold hover:bg-[#1E73BE] hover:text-white transition-all duration-300 cursor-pointer text-base transform hover:scale-105 hover:shadow-lg animate-fade-in-up delay-300 disabled:opacity-70 disabled:cursor-not-allowed"
        >
          {isSubmitting ? (
            <div className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-[#1E73BE]" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              {isSignUp ? 'Creating Account...' : 'Signing In...'}
            </div>
          ) : (
            isSignUp ? 'Create Account' : 'Sign In'
          )}
        </button>
      </form>

      {/* Divider */}
      <div className="flex items-center animate-fade-in-up delay-300">
        <div className="flex-grow border-t border-gray-200"></div>
        <span className="mx-3 text-sm font-medium text-gray-500 uppercase tracking-wide">OR CONTINUE WITH</span>
        <div className="flex-grow border-t border-gray-200"></div>
      </div>

      {/* Google button */}
      <div className="animate-fade-in-up delay-300">
        <GoogleRippleButton onClick={handleGoogleLogin} disabled={isSubmitting || isLoggedIn} />
      </div>

      {/* Footer */}
      <div className="text-center animate-fade-in-up delay-300">
        <p className="text-sm text-gray-600">
          {isSignUp ? 'Already have an account?' : "Don't have an account?"}{' '}
          <button
            onClick={() => toggleMode(isSignUp ? 'login' : 'signup')}
            className="text-[#1E73BE] font-medium hover:underline relative group cursor-pointer transition-all duration-200 transform hover:scale-105"
            disabled={isSubmitting || isLoggedIn}
          >
            {isSignUp ? 'SIGN IN' : 'SIGN UP'}
            {/* Ripple effect for the signup/login text */}
            <span className="absolute inset-0 rounded bg-[#1E73BE]/20 scale-0 opacity-0 transition-all duration-300 group-active:scale-125 group-active:opacity-100"></span>
          </button>
        </p>
      </div>
    </div>
  )
}