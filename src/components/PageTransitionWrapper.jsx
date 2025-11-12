'use client';

import { useState, useEffect } from 'react';
import { usePathname } from 'next/navigation';

export default function PageTransitionWrapper({ children }) {
  const pathname = usePathname();
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [displayChildren, setDisplayChildren] = useState(children);

  useEffect(() => {
    // When pathname changes, start exit animation
    setIsTransitioning(true);
    
    const timer = setTimeout(() => {
      setDisplayChildren(children);
      setIsTransitioning(false);
    }, 350); // Half of our transition duration

    return () => clearTimeout(timer);
  }, [pathname, children]);

  return (
    <div 
      className={`transition-all duration-700 ease-in-out ${
        isTransitioning 
          ? 'opacity-0 translate-x-4' 
          : 'opacity-100 translate-x-0'
      }`}
    >
      {displayChildren}
    </div>
  );
}