'use client';

import { FcGoogle } from 'react-icons/fc';
import { useRef } from 'react';

export function GoogleRippleButton({ onClick, className = '', disabled = false }) {
  const buttonRef = useRef(null);

  const handleClick = (e) => {
    if (disabled) return;
    
    // Create ripple element
    const ripple = document.createElement('span');
    const rect = e.currentTarget.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = e.clientX - rect.left - size / 2;
    const y = e.clientY - rect.top - size / 2;
    
    ripple.style.cssText = `
      position: absolute;
      width: ${size}px;
      height: ${size}px;
      left: ${x}px;
      top: ${y}px;
      background-color: rgba(30, 115, 190, 0.3);
      border-radius: 50%;
      transform: scale(0);
      animation: ripple 0.6s linear;
      pointer-events: none;
    `;
    
    // Add ripple to button
    buttonRef.current.appendChild(ripple);
    
    // Remove ripple after animation completes
    setTimeout(() => {
      ripple.remove();
    }, 600);
    
    // Call the provided onClick handler
    if (onClick) onClick(e);
  };

  return (
    <button
      ref={buttonRef}
      onClick={handleClick}
      disabled={disabled}
      className={`flex items-center justify-center w-full px-5 py-3 bg-white border-2 border-[#1E73BE] rounded-xl text-sm font-bold text-[#1E73BE] hover:bg-[#1E73BE] hover:text-white active:scale-95 transition-all duration-300 overflow-hidden relative cursor-pointer shadow-md hover:shadow-lg transform hover:scale-105 ${disabled ? 'opacity-70 cursor-not-allowed' : ''} ${className}`}
    >
      <div className="flex items-center justify-center w-6 h-6 rounded-md bg-white mr-3">
        <FcGoogle className="text-lg" />
      </div>
      Continue with Google
    </button>
  );
}

export default GoogleRippleButton;