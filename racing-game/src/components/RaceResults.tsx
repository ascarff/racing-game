/**
 * Race Results Component
 * Displays end-of-race screen with final time, lap breakdown, and navigation options
 */

import { useCallback, useEffect, useRef } from 'react';
import { formatTime } from '../game/gameState';
import './RaceResults.css';

interface RaceResultsProps {
  /** Whether the player won (1st place) or lost (2nd place) */
  playerPosition: 1 | 2;
  /** Total race time in milliseconds */
  totalTime: number;
  /** Array of lap times in milliseconds */
  lapTimes: number[];
  /** CPU lap times for comparison */
  cpuLapTimes: number[];
  /** CPU total time */
  cpuTotalTime: number;
  /** Callback to restart the race */
  onRaceAgain: () => void;
  /** Callback to return to main menu */
  onMainMenu: () => void;
}

/**
 * RaceResults - End of race screen showing times and options
 */
export function RaceResults({
  playerPosition,
  totalTime,
  lapTimes,
  cpuLapTimes,
  cpuTotalTime,
  onRaceAgain,
  onMainMenu,
}: RaceResultsProps) {
  const raceAgainButtonRef = useRef<HTMLButtonElement>(null);

  // Find the best lap time
  const bestLapTime = lapTimes.length > 0 ? Math.min(...lapTimes) : null;
  const bestLapIndex = bestLapTime !== null ? lapTimes.indexOf(bestLapTime) : -1;

  // Focus race again button when component mounts
  useEffect(() => {
    raceAgainButtonRef.current?.focus();
  }, []);

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (event.key === 'Enter' || event.key === ' ') {
        if (event.key === ' ') {
          event.preventDefault();
        }
        if (document.activeElement === raceAgainButtonRef.current) {
          onRaceAgain();
        }
      } else if (event.key === 'Escape') {
        onMainMenu();
      }
    },
    [onRaceAgain, onMainMenu]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);

  const isWinner = playerPosition === 1;

  return (
    <div className="race-results">
      <div className="results-content">
        {/* Race Complete Header */}
        <h1 className={`results-title ${isWinner ? 'winner' : 'loser'}`}>
          {isWinner ? 'You Win!' : 'Race Complete'}
        </h1>

        {/* Position Display */}
        <div className={`position-badge ${isWinner ? 'first-place' : 'second-place'}`}>
          {playerPosition === 1 ? '1st Place' : '2nd Place'}
        </div>

        {/* Time Comparison */}
        <div className="times-comparison">
          {/* Player Times */}
          <div className="times-column player-times">
            <h2 className="times-header player-header">Your Times</h2>
            <div className="total-time">
              <span className="time-label">Total Time</span>
              <span className="time-value">{formatTime(totalTime)}</span>
            </div>

            {/* Lap Breakdown */}
            <div className="lap-breakdown">
              <h3 className="lap-header">Lap Times</h3>
              <ul className="lap-list">
                {lapTimes.map((time, index) => (
                  <li
                    key={index}
                    className={`lap-item ${index === bestLapIndex ? 'best-lap' : ''}`}
                  >
                    <span className="lap-number">Lap {index + 1}</span>
                    <span className="lap-time">{formatTime(time)}</span>
                    {index === bestLapIndex && (
                      <span className="best-lap-badge">Best</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* CPU Times */}
          <div className="times-column cpu-times">
            <h2 className="times-header cpu-header">CPU Times</h2>
            <div className="total-time">
              <span className="time-label">Total Time</span>
              <span className="time-value">{formatTime(cpuTotalTime)}</span>
            </div>

            {/* CPU Lap Breakdown */}
            <div className="lap-breakdown">
              <h3 className="lap-header">Lap Times</h3>
              <ul className="lap-list">
                {cpuLapTimes.map((time, index) => {
                  const cpuBestLap = cpuLapTimes.length > 0 ? Math.min(...cpuLapTimes) : null;
                  const isCpuBestLap = time === cpuBestLap;
                  return (
                    <li
                      key={index}
                      className={`lap-item ${isCpuBestLap ? 'best-lap cpu-best' : ''}`}
                    >
                      <span className="lap-number">Lap {index + 1}</span>
                      <span className="lap-time">{formatTime(time)}</span>
                      {isCpuBestLap && (
                        <span className="best-lap-badge cpu-badge">Best</span>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="results-buttons">
          <button
            ref={raceAgainButtonRef}
            className="results-button race-again-button"
            onClick={onRaceAgain}
            aria-label="Race Again"
          >
            Race Again
          </button>
          <button
            className="results-button main-menu-button"
            onClick={onMainMenu}
            aria-label="Main Menu"
          >
            Main Menu
          </button>
        </div>

        <div className="results-controls">
          <p>Press <kbd>Enter</kbd> to race again or <kbd>Esc</kbd> for menu</p>
        </div>
      </div>
    </div>
  );
}

export default RaceResults;
