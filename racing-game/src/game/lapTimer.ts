/**
 * Lap Timer System for the 2D racing game
 * Tracks lap times, checkpoint progression, and validates lap completion
 */

import type { Point, Checkpoint, StartFinishLine } from './types';

/**
 * State for the lap timing system
 */
export interface LapTimerState {
  /** Current lap time in milliseconds */
  currentLapTime: number;
  /** Array of completed lap times in milliseconds */
  lapTimes: number[];
  /** Current lap number (1-indexed) */
  currentLap: number;
  /** Tracks which checkpoints have been passed this lap (indexed by checkpoint id) */
  checkpointsPassed: Map<number, boolean>;
  /** Whether the timer is currently running */
  isRunning: boolean;
  /** Total number of laps for the race (0 = unlimited/practice) */
  totalLaps: number;
  /** Whether the race has been completed */
  raceFinished: boolean;
  /** Last completed lap time for display (shows briefly after crossing finish) */
  lastLapTime: number | null;
}

/**
 * Configuration for the lap timer
 */
export interface LapTimerConfig {
  /** Total number of laps for the race (0 = unlimited) */
  totalLaps: number;
  /** Checkpoint IDs that must be passed (from track data) */
  checkpointIds: number[];
}

/**
 * Result of a lap timer update
 */
export interface LapTimerUpdateResult {
  /** Updated lap timer state */
  state: LapTimerState;
  /** Whether a new lap was just completed */
  lapCompleted: boolean;
  /** Whether a checkpoint was just crossed */
  checkpointCrossed: number | null;
}

/**
 * Create a new lap timer state
 * @param config - Configuration for the lap timer
 * @returns Initial lap timer state
 */
export function createLapTimer(config: LapTimerConfig): LapTimerState {
  const checkpointsPassed = new Map<number, boolean>();
  config.checkpointIds.forEach(id => checkpointsPassed.set(id, false));

  return {
    currentLapTime: 0,
    lapTimes: [],
    currentLap: 1,
    checkpointsPassed,
    isRunning: false,
    totalLaps: config.totalLaps,
    raceFinished: false,
    lastLapTime: null,
  };
}

/**
 * Start the lap timer
 * @param state - Current lap timer state
 * @returns Updated state with timer running
 */
export function startLapTimer(state: LapTimerState): LapTimerState {
  return {
    ...state,
    isRunning: true,
    currentLapTime: 0,
  };
}

/**
 * Pause the lap timer
 * @param state - Current lap timer state
 * @returns Updated state with timer paused
 */
export function pauseLapTimer(state: LapTimerState): LapTimerState {
  return {
    ...state,
    isRunning: false,
  };
}

/**
 * Resume the lap timer
 * @param state - Current lap timer state
 * @returns Updated state with timer running
 */
export function resumeLapTimer(state: LapTimerState): LapTimerState {
  return {
    ...state,
    isRunning: true,
  };
}

/**
 * Check if two line segments intersect
 * Uses the cross product method for line segment intersection
 * @param p1 - Start of first line segment
 * @param p2 - End of first line segment
 * @param p3 - Start of second line segment
 * @param p4 - End of second line segment
 * @returns True if the line segments intersect
 */
export function doLinesIntersect(
  p1: Point,
  p2: Point,
  p3: Point,
  p4: Point
): boolean {
  // Calculate direction vectors
  const d1x = p2.x - p1.x;
  const d1y = p2.y - p1.y;
  const d2x = p4.x - p3.x;
  const d2y = p4.y - p3.y;

  // Calculate the cross product of direction vectors
  const cross = d1x * d2y - d1y * d2x;

  // If cross product is 0, lines are parallel
  if (Math.abs(cross) < 0.0001) {
    return false;
  }

  // Calculate parameters for intersection point
  const dx = p3.x - p1.x;
  const dy = p3.y - p1.y;

  const t = (dx * d2y - dy * d2x) / cross;
  const u = (dx * d1y - dy * d1x) / cross;

  // Check if intersection point lies within both line segments
  return t >= 0 && t <= 1 && u >= 0 && u <= 1;
}

/**
 * Calculate the cross product to determine which side of a line a point is on
 * @param lineStart - Start of the line
 * @param lineEnd - End of the line
 * @param point - Point to check
 * @returns Positive if point is on left side, negative if on right side
 */
function crossProduct(lineStart: Point, lineEnd: Point, point: Point): number {
  return (lineEnd.x - lineStart.x) * (point.y - lineStart.y) -
         (lineEnd.y - lineStart.y) * (point.x - lineStart.x);
}

/**
 * Check if the car crossed a line in the correct direction (forward)
 * This prevents counting backwards crossings
 * @param lastPosition - Previous car position
 * @param currentPosition - Current car position
 * @param lineStart - Start of the line to check
 * @param lineEnd - End of the line to check
 * @returns True if crossed in the correct (counter-clockwise racing) direction
 */
export function checkCrossingDirection(
  lastPosition: Point,
  currentPosition: Point,
  lineStart: Point,
  lineEnd: Point
): boolean {
  // For our oval track, racing direction is counter-clockwise
  // The finish line is at the bottom (south), cars should cross from right to left
  // when viewed from above (from y+ going to y-)

  // Check which side of the line each position is on
  const lastSide = crossProduct(lineStart, lineEnd, lastPosition);
  const currentSide = crossProduct(lineStart, lineEnd, currentPosition);

  // For the finish line at the bottom (going from inner to outer edge),
  // a correct crossing is from positive to negative (or negative to positive
  // depending on line orientation)

  // We want to detect when the car actually crossed (signs differ)
  // and when they crossed in the "forward" direction
  // For counter-clockwise racing on our track, the car should approach
  // from the right side and exit on the left side of the finish line

  // The finish line goes from inner (top of line segment) to outer (bottom)
  // Counter-clockwise racing means crossing from right to left
  // With cross product, this means going from negative to positive
  return lastSide < 0 && currentSide >= 0;
}

/**
 * Check if car crossed a checkpoint or finish line
 * @param lastPosition - Previous car position
 * @param currentPosition - Current car position
 * @param line - The line segment to check (checkpoint or finish line)
 * @returns True if car crossed the line in the correct direction
 */
export function checkLineCrossing(
  lastPosition: Point,
  currentPosition: Point,
  line: { start: Point; end: Point }
): boolean {
  // First check if the path intersects the line at all
  if (!doLinesIntersect(lastPosition, currentPosition, line.start, line.end)) {
    return false;
  }

  // Then check if crossing was in the correct direction
  return checkCrossingDirection(lastPosition, currentPosition, line.start, line.end);
}

/**
 * Check if all checkpoints have been passed this lap
 * @param checkpointsPassed - Map of checkpoint IDs to passed status
 * @returns True if all checkpoints have been passed
 */
export function allCheckpointsPassed(checkpointsPassed: Map<number, boolean>): boolean {
  for (const passed of checkpointsPassed.values()) {
    if (!passed) return false;
  }
  return true;
}

/**
 * Reset checkpoint tracking for a new lap
 * @param checkpointsPassed - Current checkpoint map
 * @returns New map with all checkpoints reset to false
 */
function resetCheckpoints(checkpointsPassed: Map<number, boolean>): Map<number, boolean> {
  const newMap = new Map<number, boolean>();
  for (const id of checkpointsPassed.keys()) {
    newMap.set(id, false);
  }
  return newMap;
}

/**
 * Update the lap timer state
 * Call this every frame to track time and detect line crossings
 *
 * @param state - Current lap timer state
 * @param carPosition - Current car position
 * @param lastPosition - Car position from previous frame
 * @param deltaTime - Time since last frame in seconds
 * @param checkpoints - Array of track checkpoints
 * @param startFinishLine - The start/finish line
 * @returns Updated lap timer state and any events that occurred
 */
export function updateLapTimer(
  state: LapTimerState,
  carPosition: Point,
  lastPosition: Point,
  deltaTime: number,
  checkpoints: Checkpoint[],
  startFinishLine: StartFinishLine
): LapTimerUpdateResult {
  // If timer is not running or race is finished, return unchanged state
  if (!state.isRunning || state.raceFinished) {
    return { state, lapCompleted: false, checkpointCrossed: null };
  }

  // Update current lap time
  let newCurrentLapTime = state.currentLapTime + deltaTime * 1000;
  let newCheckpointsPassed = state.checkpointsPassed;
  let newLapTimes = state.lapTimes;
  let newCurrentLap = state.currentLap;
  let newRaceFinished: boolean = state.raceFinished;
  let newLastLapTime = state.lastLapTime;
  let lapCompleted = false;
  let checkpointCrossed: number | null = null;

  // Check for checkpoint crossings
  for (const checkpoint of checkpoints) {
    if (!state.checkpointsPassed.get(checkpoint.id)) {
      if (checkLineCrossing(lastPosition, carPosition, checkpoint.line)) {
        newCheckpointsPassed = new Map(state.checkpointsPassed);
        newCheckpointsPassed.set(checkpoint.id, true);
        checkpointCrossed = checkpoint.id;
        break; // Only count one checkpoint per frame
      }
    }
  }

  // Check for finish line crossing
  if (checkLineCrossing(lastPosition, carPosition, startFinishLine)) {
    // Only count the lap if all checkpoints have been passed
    const checkpointsToCheck = checkpointCrossed !== null ? newCheckpointsPassed : state.checkpointsPassed;

    if (allCheckpointsPassed(checkpointsToCheck)) {
      // Valid lap completed!
      lapCompleted = true;
      newLastLapTime = newCurrentLapTime;
      newLapTimes = [...state.lapTimes, newCurrentLapTime];
      newCurrentLap = state.currentLap + 1;
      newCurrentLapTime = 0;

      // Reset checkpoints for next lap
      newCheckpointsPassed = resetCheckpoints(state.checkpointsPassed);

      // Check if race is finished
      if (state.totalLaps > 0 && newLapTimes.length >= state.totalLaps) {
        newRaceFinished = true;
      }
    }
  }

  return {
    state: {
      ...state,
      currentLapTime: newCurrentLapTime,
      lapTimes: newLapTimes,
      currentLap: newCurrentLap,
      checkpointsPassed: newCheckpointsPassed,
      raceFinished: newRaceFinished,
      lastLapTime: newLastLapTime,
    },
    lapCompleted,
    checkpointCrossed,
  };
}

/**
 * Get the best lap time from completed laps
 * @param state - Current lap timer state
 * @returns Best lap time in milliseconds, or null if no laps completed
 */
export function getBestLapTime(state: LapTimerState): number | null {
  if (state.lapTimes.length === 0) return null;
  return Math.min(...state.lapTimes);
}

/**
 * Get the total race time (sum of all completed lap times plus current lap time)
 * @param state - Current lap timer state
 * @returns Total time in milliseconds
 */
export function getTotalRaceTime(state: LapTimerState): number {
  const completedTime = state.lapTimes.reduce((sum, time) => sum + time, 0);
  return completedTime + state.currentLapTime;
}

/**
 * Reset the lap timer to initial state
 * @param state - Current lap timer state
 * @returns Reset lap timer state
 */
export function resetLapTimer(state: LapTimerState): LapTimerState {
  return {
    ...state,
    currentLapTime: 0,
    lapTimes: [],
    currentLap: 1,
    checkpointsPassed: resetCheckpoints(state.checkpointsPassed),
    isRunning: false,
    raceFinished: false,
    lastLapTime: null,
  };
}
