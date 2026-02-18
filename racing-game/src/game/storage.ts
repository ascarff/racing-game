/**
 * Best Times Persistence
 * Handles saving and loading best times to/from localStorage
 */

/** Key used to store best times in localStorage */
export const STORAGE_KEY = 'racing-game-best-times';

/**
 * Structure for persisted best times
 */
export interface BestTimes {
  /** Best single lap time in milliseconds */
  bestLapTime: number | null;
  /** Best total race time in milliseconds (sum of all laps) */
  bestRaceTime: number | null;
}

/**
 * Default best times when no data is stored
 */
const DEFAULT_BEST_TIMES: BestTimes = {
  bestLapTime: null,
  bestRaceTime: null,
};

/**
 * Save best times to localStorage
 * @param times - The best times to save
 */
export function saveBestTimes(times: BestTimes): void {
  try {
    const serialized = JSON.stringify(times);
    localStorage.setItem(STORAGE_KEY, serialized);
  } catch (error) {
    // localStorage might be unavailable (e.g., private browsing mode)
    console.warn('Failed to save best times to localStorage:', error);
  }
}

/**
 * Load best times from localStorage
 * @returns The stored best times, or default values if none exist
 */
export function loadBestTimes(): BestTimes {
  try {
    const serialized = localStorage.getItem(STORAGE_KEY);
    if (serialized === null) {
      return { ...DEFAULT_BEST_TIMES };
    }

    const parsed = JSON.parse(serialized);

    // Validate the loaded data has expected shape
    if (typeof parsed !== 'object' || parsed === null) {
      return { ...DEFAULT_BEST_TIMES };
    }

    // Return valid data with defaults for missing fields
    return {
      bestLapTime: typeof parsed.bestLapTime === 'number' ? parsed.bestLapTime : null,
      bestRaceTime: typeof parsed.bestRaceTime === 'number' ? parsed.bestRaceTime : null,
    };
  } catch (error) {
    // localStorage might be unavailable or data might be corrupted
    console.warn('Failed to load best times from localStorage:', error);
    return { ...DEFAULT_BEST_TIMES };
  }
}

/**
 * Clear all best times from localStorage
 */
export function clearBestTimes(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (error) {
    console.warn('Failed to clear best times from localStorage:', error);
  }
}

/**
 * Update best times if new time is better
 * @param currentBest - Current best times
 * @param newLapTime - New lap time to compare (optional)
 * @param newRaceTime - New race time to compare (optional)
 * @returns Updated best times and flags indicating if records were broken
 */
export function updateBestTimes(
  currentBest: BestTimes,
  newLapTime?: number,
  newRaceTime?: number
): { times: BestTimes; newLapRecord: boolean; newRaceRecord: boolean } {
  let newLapRecord = false;
  let newRaceRecord = false;

  const updatedTimes: BestTimes = { ...currentBest };

  // Check if new lap time is a record
  if (newLapTime !== undefined) {
    if (currentBest.bestLapTime === null || newLapTime < currentBest.bestLapTime) {
      updatedTimes.bestLapTime = newLapTime;
      newLapRecord = true;
    }
  }

  // Check if new race time is a record
  if (newRaceTime !== undefined) {
    if (currentBest.bestRaceTime === null || newRaceTime < currentBest.bestRaceTime) {
      updatedTimes.bestRaceTime = newRaceTime;
      newRaceRecord = true;
    }
  }

  return { times: updatedTimes, newLapRecord, newRaceRecord };
}
