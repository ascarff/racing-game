/**
 * Main Menu Component
 * Displays the game title, best time, and start button
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { formatTime } from '../game/gameState';
import './MainMenu.css';

interface MainMenuProps {
  /** Best time in milliseconds (null if no best time) */
  bestTime: number | null;
  /** Callback to start the race */
  onStartRace: () => void;
  /** Callback to clear best times */
  onClearBestTimes: () => void;
}

/**
 * MainMenu - The game's main menu screen
 */
export function MainMenu({ bestTime, onStartRace, onClearBestTimes }: MainMenuProps) {
  const startButtonRef = useRef<HTMLButtonElement>(null);
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  // Focus the start button when menu mounts for keyboard accessibility
  useEffect(() => {
    startButtonRef.current?.focus();
  }, []);

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (event.key === 'Enter' || event.key === ' ') {
        // Prevent default space scrolling
        if (event.key === ' ') {
          event.preventDefault();
        }
        // Only trigger if focus is on start button or document body
        if (
          document.activeElement === startButtonRef.current ||
          document.activeElement === document.body
        ) {
          onStartRace();
        }
      }
    },
    [onStartRace]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);

  return (
    <div className="main-menu">
      <div className="menu-content">
        <h1 className="menu-title">2D Racing</h1>
        <p className="menu-subtitle">Time Trial</p>

        <div className="best-time-container">
          <span className="best-time-label">Best Lap Time</span>
          <span className="best-time-value">{formatTime(bestTime)}</span>
          {bestTime !== null && (
            <button
              className="clear-times-button"
              onClick={() => setShowClearConfirm(true)}
              aria-label="Clear Best Times"
            >
              Clear
            </button>
          )}
        </div>

        {/* Clear confirmation dialog */}
        {showClearConfirm && (
          <div className="confirm-dialog">
            <p>Clear all best times?</p>
            <div className="confirm-buttons">
              <button
                className="confirm-yes"
                onClick={() => {
                  onClearBestTimes();
                  setShowClearConfirm(false);
                }}
              >
                Yes, Clear
              </button>
              <button
                className="confirm-no"
                onClick={() => setShowClearConfirm(false)}
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        <button
          ref={startButtonRef}
          className="start-button"
          onClick={onStartRace}
          aria-label="Start Race"
        >
          Start Race
        </button>

        <div className="menu-controls">
          <p>Press <kbd>Enter</kbd> or click to start</p>
        </div>
      </div>
    </div>
  );
}

export default MainMenu;
