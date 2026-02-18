/**
 * Countdown Component
 * Displays a 3-2-1-GO! countdown before the race starts
 * Player cannot move during countdown
 */

import { useState, useEffect } from 'react';
import './Countdown.css';

/**
 * Countdown state type
 * null = countdown finished, number = displaying that number, 'GO' = displaying GO!
 */
type CountdownState = 3 | 2 | 1 | 'GO' | null;

interface CountdownProps {
  /** Whether to start the countdown */
  isActive: boolean;
  /** Callback when countdown completes */
  onComplete: () => void;
}

/**
 * Countdown component - shows 3, 2, 1, GO! before race starts
 */
export function Countdown({ isActive, onComplete }: CountdownProps) {
  const [countdownState, setCountdownState] = useState<CountdownState>(null);

  useEffect(() => {
    if (!isActive) {
      setCountdownState(null);
      return;
    }

    // Start countdown from 3
    setCountdownState(3);

    // Schedule countdown transitions
    const timers: ReturnType<typeof setTimeout>[] = [];

    // 3 -> 2 after 1 second
    timers.push(setTimeout(() => setCountdownState(2), 1000));

    // 2 -> 1 after 2 seconds
    timers.push(setTimeout(() => setCountdownState(1), 2000));

    // 1 -> GO! after 3 seconds
    timers.push(setTimeout(() => setCountdownState('GO'), 3000));

    // GO! -> null (finished) after 3.5 seconds and call onComplete
    timers.push(setTimeout(() => {
      setCountdownState(null);
      onComplete();
    }, 3500));

    // Cleanup timers on unmount or when isActive changes
    return () => {
      timers.forEach(timer => clearTimeout(timer));
    };
  }, [isActive, onComplete]);

  // Don't render anything if countdown is not active or finished
  if (countdownState === null) {
    return null;
  }

  const displayText = countdownState === 'GO' ? 'GO!' : countdownState.toString();
  const isGo = countdownState === 'GO';

  return (
    <div className="countdown-overlay">
      <div className={`countdown-text ${isGo ? 'countdown-go' : 'countdown-number'}`}>
        {displayText}
      </div>
    </div>
  );
}

export default Countdown;
