/**
 * CPU Driver AI System for the 2D racing game
 * Provides path-following AI opponent using waypoints derived from track center line
 */

import type { Point } from './types';
import type { InputState } from './input';
import {
  createCarState,
  updateCar,
  createInputFromDirection,
  type CarState,
  type CarConfig,
  defaultCarConfig,
} from './car';
import { getTrackCenterPoint } from './track';

/**
 * Configuration for the CPU driver AI
 */
export interface CPUDriverConfig {
  /** Number of waypoints around the track */
  numWaypoints: number;
  /** Distance threshold to switch to next waypoint (pixels) */
  waypointThreshold: number;
  /** How far ahead to look for upcoming turns (in waypoint indices) */
  lookAheadWaypoints: number;
  /** Turn angle threshold for braking (radians) */
  turnBrakeThreshold: number;
  /** Speed multiplier when approaching sharp turns (0-1) */
  turnSpeedFactor: number;
  /** Offset from center line (negative = inside, positive = outside) */
  racingLineOffset: number;
}

/**
 * Default CPU driver configuration
 * Tuned for competitive but beatable AI
 */
export const defaultCPUDriverConfig: CPUDriverConfig = {
  numWaypoints: 32,
  waypointThreshold: 40,
  lookAheadWaypoints: 4,
  turnBrakeThreshold: Math.PI / 6, // 30 degrees
  turnSpeedFactor: 0.6,
  racingLineOffset: 0,
};

/**
 * CPU Driver state
 * Contains car state and AI decision-making state
 */
export interface CPUDriver {
  /** The car physics state */
  carState: CarState;
  /** Index of the current target waypoint */
  targetWaypoint: number;
  /** Array of waypoints around the track */
  waypoints: Point[];
  /** AI configuration */
  config: CPUDriverConfig;
  /** Current steering decision (-1 to 1, negative = left, positive = right) */
  currentSteering: number;
  /** Current throttle decision (true = accelerate, false = coast/brake) */
  currentThrottle: boolean;
  /** Current brake decision */
  currentBrake: boolean;
  /** Last position for collision detection */
  lastPosition: Point;
}

/**
 * Generate waypoints around the track centerline
 * @param numWaypoints - Number of waypoints to generate
 * @param offset - Offset from centerline (for racing line variation)
 * @returns Array of waypoint positions
 */
export function generateWaypoints(numWaypoints: number, offset: number = 0): Point[] {
  const waypoints: Point[] = [];

  for (let i = 0; i < numWaypoints; i++) {
    // Calculate angle around the track (0 to 2*PI)
    // Start at the bottom (PI/2) and go counter-clockwise
    const angle = (Math.PI / 2) + (i / numWaypoints) * Math.PI * 2;
    const centerPoint = getTrackCenterPoint(angle);

    // Apply racing line offset if specified
    // Offset moves the waypoint toward inside (negative) or outside (positive)
    if (offset !== 0) {
      // Calculate normal direction (toward center of ellipse)
      const trackCenterX = 600; // CANVAS_WIDTH / 2
      const trackCenterY = 400; // CANVAS_HEIGHT / 2
      const dx = centerPoint.x - trackCenterX;
      const dy = centerPoint.y - trackCenterY;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist > 0) {
        // Normal pointing outward from track center
        const nx = dx / dist;
        const ny = dy / dist;
        centerPoint.x += nx * offset;
        centerPoint.y += ny * offset;
      }
    }

    waypoints.push(centerPoint);
  }

  return waypoints;
}

/**
 * Calculate the angle from the car to a target point
 * @param carPosition - Current car position
 * @param carAngle - Current car angle
 * @param target - Target point
 * @returns Angle difference in radians (-PI to PI)
 */
function calculateAngleToTarget(carPosition: Point, carAngle: number, target: Point): number {
  const dx = target.x - carPosition.x;
  const dy = target.y - carPosition.y;
  const targetAngle = Math.atan2(dy, dx);

  // Calculate angle difference
  let angleDiff = targetAngle - carAngle;

  // Normalize to -PI to PI range
  while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
  while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;

  return angleDiff;
}

/**
 * Calculate distance between two points
 */
function distance(p1: Point, p2: Point): number {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Calculate the sharpness of upcoming turns
 * @param waypoints - Array of waypoints
 * @param currentIndex - Current waypoint index
 * @param lookAhead - How many waypoints ahead to check
 * @returns Maximum turn angle in radians
 */
function calculateUpcomingTurnSharpness(
  waypoints: Point[],
  currentIndex: number,
  lookAhead: number
): number {
  let maxTurnAngle = 0;
  const numWaypoints = waypoints.length;

  for (let i = 0; i < lookAhead - 1; i++) {
    const wp1 = waypoints[(currentIndex + i) % numWaypoints];
    const wp2 = waypoints[(currentIndex + i + 1) % numWaypoints];
    const wp3 = waypoints[(currentIndex + i + 2) % numWaypoints];

    // Calculate angle at wp2
    const angle1 = Math.atan2(wp2.y - wp1.y, wp2.x - wp1.x);
    const angle2 = Math.atan2(wp3.y - wp2.y, wp3.x - wp2.x);

    let turnAngle = Math.abs(angle2 - angle1);
    if (turnAngle > Math.PI) turnAngle = Math.PI * 2 - turnAngle;

    maxTurnAngle = Math.max(maxTurnAngle, turnAngle);
  }

  return maxTurnAngle;
}

/**
 * Create a CPU driver at a given position
 * @param position - Starting position
 * @param angle - Starting angle
 * @param carConfig - Optional car physics configuration
 * @param driverConfig - Optional AI configuration
 * @returns New CPU driver state
 */
export function createCPUDriver(
  position: Point,
  angle: number,
  carConfig?: Partial<CarConfig>,
  driverConfig?: Partial<CPUDriverConfig>
): CPUDriver {
  // Create CPU car config - slightly slower than player for fairness
  const cpuCarConfig: CarConfig = {
    ...defaultCarConfig,
    maxSpeed: 320, // Slightly slower than player's 350
    acceleration: 230, // Slightly less acceleration
    ...carConfig,
  };

  const config: CPUDriverConfig = {
    ...defaultCPUDriverConfig,
    ...driverConfig,
  };

  const waypoints = generateWaypoints(config.numWaypoints, config.racingLineOffset);

  // Find the closest waypoint to start position
  let closestWaypoint = 0;
  let closestDistance = Infinity;

  for (let i = 0; i < waypoints.length; i++) {
    const dist = distance(position, waypoints[i]);
    if (dist < closestDistance) {
      closestDistance = dist;
      closestWaypoint = i;
    }
  }

  // Start targeting the next waypoint (ahead of current position)
  const targetWaypoint = (closestWaypoint + 1) % waypoints.length;

  return {
    carState: createCarState(position, angle, cpuCarConfig),
    targetWaypoint,
    waypoints,
    config,
    currentSteering: 0,
    currentThrottle: true,
    currentBrake: false,
    lastPosition: { ...position },
  };
}

/**
 * Generate AI input based on current state
 * This is the main AI decision-making function
 * @param driver - CPU driver state
 * @returns InputState for the car physics system
 */
function generateAIInput(driver: CPUDriver): InputState {
  const { carState, waypoints, targetWaypoint, config } = driver;
  const car = carState;

  // Get target waypoint
  const target = waypoints[targetWaypoint];

  // Calculate angle to target
  const angleToTarget = calculateAngleToTarget(car.position, car.angle, target);

  // --- STEERING DECISION ---
  // Use proportional steering based on angle difference
  const steeringThreshold = 0.05; // Dead zone to prevent jitter
  let steerLeft = false;
  let steerRight = false;

  if (angleToTarget < -steeringThreshold) {
    steerLeft = true;
  } else if (angleToTarget > steeringThreshold) {
    steerRight = true;
  }

  // --- THROTTLE/BRAKE DECISION ---
  // Check upcoming turn sharpness
  const upcomingTurnAngle = calculateUpcomingTurnSharpness(
    waypoints,
    targetWaypoint,
    config.lookAheadWaypoints
  );

  // Determine if we need to slow down for turns
  const needsBraking = upcomingTurnAngle > config.turnBrakeThreshold;
  const targetSpeed = needsBraking
    ? car.config.maxSpeed * config.turnSpeedFactor
    : car.config.maxSpeed;

  let accelerate = false;
  let brake = false;

  if (car.speed < targetSpeed * 0.95) {
    // Need to speed up
    accelerate = true;
  } else if (car.speed > targetSpeed * 1.1 && needsBraking) {
    // Going too fast into a turn, brake
    brake = true;
  } else if (!needsBraking) {
    // On a straight, always accelerate
    accelerate = true;
  }

  // If colliding, back off throttle
  if (car.isColliding) {
    accelerate = false;
    brake = true;
  }

  // Store current decisions for debugging
  driver.currentSteering = angleToTarget;
  driver.currentThrottle = accelerate;
  driver.currentBrake = brake;

  return createInputFromDirection(accelerate, brake, steerLeft, steerRight);
}

/**
 * Update the CPU driver for one frame
 * Handles AI decision making and car physics
 * @param driver - CPU driver state (will be mutated)
 * @param deltaTime - Time since last frame in seconds
 */
export function updateCPUDriver(driver: CPUDriver, deltaTime: number): void {
  // Store last position for lap tracking
  driver.lastPosition = { ...driver.carState.position };

  // Generate AI input
  const input = generateAIInput(driver);

  // Update car physics
  updateCar(driver.carState, input, deltaTime);

  // Check if we've reached the target waypoint
  const distanceToTarget = distance(
    driver.carState.position,
    driver.waypoints[driver.targetWaypoint]
  );

  if (distanceToTarget < driver.config.waypointThreshold) {
    // Move to next waypoint
    driver.targetWaypoint = (driver.targetWaypoint + 1) % driver.waypoints.length;
  }
}

/**
 * Get CPU car state for rendering and collision detection
 * @param driver - CPU driver state
 * @returns Car state
 */
export function getCPUCarState(driver: CPUDriver): CarState {
  return driver.carState;
}

/**
 * Get the current target waypoint for debugging
 * @param driver - CPU driver state
 * @returns Target waypoint position
 */
export function getTargetWaypoint(driver: CPUDriver): Point {
  return driver.waypoints[driver.targetWaypoint];
}

/**
 * Reset CPU driver to starting position
 * @param driver - CPU driver state
 * @param position - New starting position
 * @param angle - New starting angle
 */
export function resetCPUDriver(driver: CPUDriver, position: Point, angle: number): void {
  driver.carState.position = { ...position };
  driver.carState.angle = angle;
  driver.carState.velocity = { x: 0, y: 0 };
  driver.carState.speed = 0;
  driver.carState.isColliding = false;
  driver.lastPosition = { ...position };

  // Find closest waypoint to restart from
  let closestWaypoint = 0;
  let closestDistance = Infinity;

  for (let i = 0; i < driver.waypoints.length; i++) {
    const dist = distance(position, driver.waypoints[i]);
    if (dist < closestDistance) {
      closestDistance = dist;
      closestWaypoint = i;
    }
  }

  driver.targetWaypoint = (closestWaypoint + 1) % driver.waypoints.length;
}

/**
 * Draw waypoints for debugging
 * @param ctx - Canvas rendering context
 * @param driver - CPU driver state
 */
export function drawWaypoints(
  ctx: CanvasRenderingContext2D,
  driver: CPUDriver
): void {
  const { waypoints, targetWaypoint } = driver;

  ctx.save();

  // Draw all waypoints as small dots
  ctx.fillStyle = '#4488ff44';
  for (let i = 0; i < waypoints.length; i++) {
    const wp = waypoints[i];
    ctx.beginPath();
    ctx.arc(wp.x, wp.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  // Draw target waypoint larger and brighter
  ctx.fillStyle = '#44ff44';
  const target = waypoints[targetWaypoint];
  ctx.beginPath();
  ctx.arc(target.x, target.y, 8, 0, Math.PI * 2);
  ctx.fill();

  // Draw line from car to target
  ctx.strokeStyle = '#44ff4488';
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(driver.carState.position.x, driver.carState.position.y);
  ctx.lineTo(target.x, target.y);
  ctx.stroke();

  ctx.restore();
}
