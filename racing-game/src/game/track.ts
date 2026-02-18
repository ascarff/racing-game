/**
 * Track definition and data for the 2D racing game
 * Creates an oval circuit with turns
 */

import type {
  Point,
  TrackData,
  TrackBoundary,
  StartFinishLine,
  Checkpoint,
  TrackRenderConfig,
} from './types';

// Track dimensions
const CANVAS_WIDTH = 1200;
const CANVAS_HEIGHT = 800;
const TRACK_WIDTH = 100;

// Track center and radii for the oval
const CENTER_X = CANVAS_WIDTH / 2;
const CENTER_Y = CANVAS_HEIGHT / 2;
const OUTER_RADIUS_X = 450;
const OUTER_RADIUS_Y = 300;
const INNER_RADIUS_X = OUTER_RADIUS_X - TRACK_WIDTH;
const INNER_RADIUS_Y = OUTER_RADIUS_Y - TRACK_WIDTH;

/**
 * Generate points along an ellipse
 * @param centerX - X coordinate of ellipse center
 * @param centerY - Y coordinate of ellipse center
 * @param radiusX - Horizontal radius
 * @param radiusY - Vertical radius
 * @param numPoints - Number of points to generate
 * @returns Array of points along the ellipse
 */
function generateEllipsePoints(
  centerX: number,
  centerY: number,
  radiusX: number,
  radiusY: number,
  numPoints: number
): Point[] {
  const points: Point[] = [];
  for (let i = 0; i < numPoints; i++) {
    const angle = (i / numPoints) * Math.PI * 2;
    points.push({
      x: centerX + Math.cos(angle) * radiusX,
      y: centerY + Math.sin(angle) * radiusY,
    });
  }
  return points;
}

/**
 * Create track boundaries (inner and outer edges)
 */
function createTrackBoundaries(): TrackBoundary {
  const numPoints = 100; // Smooth curves

  return {
    outer: generateEllipsePoints(CENTER_X, CENTER_Y, OUTER_RADIUS_X, OUTER_RADIUS_Y, numPoints),
    inner: generateEllipsePoints(CENTER_X, CENTER_Y, INNER_RADIUS_X, INNER_RADIUS_Y, numPoints),
  };
}

/**
 * Create the start/finish line
 * Positioned at the bottom of the oval (south position)
 */
function createStartFinishLine(): StartFinishLine {
  const startX = CENTER_X;
  const innerY = CENTER_Y + INNER_RADIUS_Y;
  const outerY = CENTER_Y + OUTER_RADIUS_Y;

  return {
    start: { x: startX, y: innerY },
    end: { x: startX, y: outerY },
    startAngle: -Math.PI / 2, // Facing up (counter-clockwise racing direction)
  };
}

/**
 * Create checkpoints around the track for lap validation
 * Positioned at key points: top, left, right
 */
function createCheckpoints(): Checkpoint[] {
  return [
    // Top checkpoint
    {
      id: 1,
      line: {
        start: { x: CENTER_X, y: CENTER_Y - INNER_RADIUS_Y },
        end: { x: CENTER_X, y: CENTER_Y - OUTER_RADIUS_Y },
      },
    },
    // Left checkpoint (west)
    {
      id: 2,
      line: {
        start: { x: CENTER_X - INNER_RADIUS_X, y: CENTER_Y },
        end: { x: CENTER_X - OUTER_RADIUS_X, y: CENTER_Y },
      },
    },
    // Right checkpoint (east)
    {
      id: 3,
      line: {
        start: { x: CENTER_X + INNER_RADIUS_X, y: CENTER_Y },
        end: { x: CENTER_X + OUTER_RADIUS_X, y: CENTER_Y },
      },
    },
  ];
}

/**
 * Default rendering configuration for the track
 */
export const defaultTrackRenderConfig: TrackRenderConfig = {
  trackColor: '#4a4a4a', // Dark gray asphalt
  boundaryColor: '#ffffff', // White boundary lines
  boundaryWidth: 4,
  startFinishColor: '#ff0000', // Red start/finish line
  startFinishWidth: 8,
  backgroundColor: '#1a472a', // Dark green grass
  grassColor: '#2d5a3d', // Slightly lighter green for variety
};

/**
 * Complete track data for the racing game
 * Export this for use by other components (e.g., car physics, collision detection)
 */
export const trackData: TrackData = {
  boundaries: createTrackBoundaries(),
  startFinishLine: createStartFinishLine(),
  checkpoints: createCheckpoints(),
  trackWidth: TRACK_WIDTH,
  dimensions: {
    width: CANVAS_WIDTH,
    height: CANVAS_HEIGHT,
  },
  // Starting position on the track (middle of start/finish line)
  startPosition: {
    x: CENTER_X,
    y: CENTER_Y + (INNER_RADIUS_Y + OUTER_RADIUS_Y) / 2,
  },
  startAngle: -Math.PI / 2, // Facing up
};

/**
 * Get a point on the track center line at a given angle
 * Useful for AI or positioning
 * @param angle - Angle in radians (0 = right, PI/2 = bottom)
 * @returns Point on the track center line
 */
export function getTrackCenterPoint(angle: number): Point {
  const centerRadiusX = (INNER_RADIUS_X + OUTER_RADIUS_X) / 2;
  const centerRadiusY = (INNER_RADIUS_Y + OUTER_RADIUS_Y) / 2;

  return {
    x: CENTER_X + Math.cos(angle) * centerRadiusX,
    y: CENTER_Y + Math.sin(angle) * centerRadiusY,
  };
}

/**
 * Check if a point is on the track (between inner and outer boundaries)
 * Simplified check using ellipse math
 * @param point - Point to check
 * @returns true if point is on the track
 */
export function isPointOnTrack(point: Point): boolean {
  // Normalize point relative to center
  const dx = point.x - CENTER_X;
  const dy = point.y - CENTER_Y;

  // Check if point is inside outer ellipse
  const outerCheck = (dx * dx) / (OUTER_RADIUS_X * OUTER_RADIUS_X) +
                     (dy * dy) / (OUTER_RADIUS_Y * OUTER_RADIUS_Y);

  // Check if point is outside inner ellipse
  const innerCheck = (dx * dx) / (INNER_RADIUS_X * INNER_RADIUS_X) +
                     (dy * dy) / (INNER_RADIUS_Y * INNER_RADIUS_Y);

  return outerCheck <= 1 && innerCheck >= 1;
}

/**
 * Get the track normal (direction perpendicular to track) at a point
 * Useful for collision response
 * @param point - Point on or near the track
 * @returns Normalized direction vector pointing toward track center
 */
export function getTrackNormal(point: Point): Point {
  const dx = point.x - CENTER_X;
  const dy = point.y - CENTER_Y;
  const length = Math.sqrt(dx * dx + dy * dy);

  if (length === 0) return { x: 0, y: -1 };

  return {
    x: -dx / length,
    y: -dy / length,
  };
}
