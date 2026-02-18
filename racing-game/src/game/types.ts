/**
 * Shared game types for the 2D racing game
 */

/**
 * Represents a 2D point in the game world
 */
export interface Point {
  x: number;
  y: number;
}

/**
 * Represents a line segment between two points
 */
export interface LineSegment {
  start: Point;
  end: Point;
}

/**
 * Track boundary definition
 * Contains inner and outer boundaries of the track
 */
export interface TrackBoundary {
  /** Points defining the inner edge of the track */
  inner: Point[];
  /** Points defining the outer edge of the track */
  outer: Point[];
}

/**
 * Start/finish line definition
 */
export interface StartFinishLine {
  /** Starting point of the line */
  start: Point;
  /** Ending point of the line */
  end: Point;
  /** Direction angle in radians for cars starting here */
  startAngle: number;
}

/**
 * Checkpoint for lap tracking and position validation
 */
export interface Checkpoint {
  /** Unique identifier for the checkpoint */
  id: number;
  /** Line segment defining the checkpoint */
  line: LineSegment;
}

/**
 * Complete track data exported for use by other game components
 */
export interface TrackData {
  /** Track boundaries (inner and outer edges) */
  boundaries: TrackBoundary;
  /** Start/finish line location and orientation */
  startFinishLine: StartFinishLine;
  /** Checkpoints for lap validation */
  checkpoints: Checkpoint[];
  /** Track width (distance between inner and outer boundaries) */
  trackWidth: number;
  /** Canvas/track dimensions */
  dimensions: {
    width: number;
    height: number;
  };
  /** Starting position for vehicles */
  startPosition: Point;
  /** Starting angle for vehicles (radians) */
  startAngle: number;
}

/**
 * Rendering configuration for the track
 */
export interface TrackRenderConfig {
  /** Track surface color */
  trackColor: string;
  /** Boundary line color */
  boundaryColor: string;
  /** Boundary line width */
  boundaryWidth: number;
  /** Start/finish line color */
  startFinishColor: string;
  /** Start/finish line width */
  startFinishWidth: number;
  /** Background color (off-track area) */
  backgroundColor: string;
  /** Grass/terrain color */
  grassColor: string;
}

/**
 * Game state for the racing game
 */
export interface GameState {
  /** Current timestamp */
  timestamp: number;
  /** Delta time since last frame (seconds) */
  deltaTime: number;
  /** Whether the game is currently running */
  isRunning: boolean;
}

/**
 * Vehicle state (for future Story 2 integration)
 */
export interface VehicleState {
  /** Current position */
  position: Point;
  /** Current rotation angle in radians */
  angle: number;
  /** Current velocity */
  velocity: Point;
  /** Current speed (magnitude of velocity) */
  speed: number;
}
