/**
 * Game State Management
 * Handles screen transitions and global game state
 */

/**
 * Possible screens/states in the game
 */
export type GameScreen = 'menu' | 'racing' | 'results';

/**
 * Global application state
 */
export interface AppState {
  /** Current screen being displayed */
  currentScreen: GameScreen;
  /** Best lap time in milliseconds (null if no best time recorded) */
  bestTime: number | null;
  /** Current race time in milliseconds (while racing) */
  currentRaceTime: number | null;
}

/**
 * Initial state when the app loads
 */
export const initialAppState: AppState = {
  currentScreen: 'menu',
  bestTime: null,
  currentRaceTime: null,
};

/**
 * Format a time in milliseconds to MM:SS:mmm format
 * @param timeMs Time in milliseconds
 * @returns Formatted time string
 */
export function formatTime(timeMs: number | null): string {
  if (timeMs === null) {
    return '--:--.---';
  }

  const totalSeconds = Math.floor(timeMs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const milliseconds = timeMs % 1000;

  const minStr = minutes.toString().padStart(2, '0');
  const secStr = seconds.toString().padStart(2, '0');
  const msStr = milliseconds.toString().padStart(3, '0');

  return `${minStr}:${secStr}.${msStr}`;
}

/**
 * Action types for state transitions
 */
export type GameAction =
  | { type: 'START_RACE' }
  | { type: 'FINISH_RACE'; time: number }
  | { type: 'RETURN_TO_MENU' }
  | { type: 'UPDATE_BEST_TIME'; time: number };

/**
 * Reducer function for game state transitions
 * @param state Current state
 * @param action Action to perform
 * @returns New state
 */
export function gameStateReducer(state: AppState, action: GameAction): AppState {
  switch (action.type) {
    case 'START_RACE':
      return {
        ...state,
        currentScreen: 'racing',
        currentRaceTime: 0,
      };

    case 'FINISH_RACE':
      return {
        ...state,
        currentScreen: 'results',
        currentRaceTime: action.time,
        bestTime: state.bestTime === null
          ? action.time
          : Math.min(state.bestTime, action.time),
      };

    case 'RETURN_TO_MENU':
      return {
        ...state,
        currentScreen: 'menu',
        currentRaceTime: null,
      };

    case 'UPDATE_BEST_TIME':
      return {
        ...state,
        bestTime: state.bestTime === null
          ? action.time
          : Math.min(state.bestTime, action.time),
      };

    default:
      return state;
  }
}
