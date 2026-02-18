/**
 * Car physics system for the 2D racing game
 * Handles acceleration, steering, momentum, and collision detection
 */

import type { Point, VehicleState } from './types';
import type { InputState } from './input';
import { isPointOnTrack, getTrackNormal } from './track';

/**
 * Car physics configuration
 * Tunable parameters for car behavior
 */
export interface CarConfig {
  /** Maximum forward speed in pixels per second */
  maxSpeed: number;
  /** Maximum reverse speed in pixels per second */
  maxReverseSpeed: number;
  /** Acceleration rate in pixels per second squared */
  acceleration: number;
  /** Braking deceleration rate in pixels per second squared */
  braking: number;
  /** Natural friction/drag deceleration when not accelerating */
  friction: number;
  /** Base turn rate in radians per second at low speed */
  baseTurnRate: number;
  /** Turn rate multiplier based on speed (higher = less turning at high speed) */
  turnSpeedFactor: number;
  /** Minimum speed required to turn (prevents spinning in place) */
  minSpeedToTurn: number;
  /** Speed reduction on collision (0-1, where 0.5 = lose half speed) */
  collisionSpeedLoss: number;
  /** How strongly the car is pushed back onto track on collision */
  collisionPushStrength: number;
  /** Car hitbox radius for collision detection */
  collisionRadius: number;
}

/**
 * Default car configuration
 * Balanced for arcade-style gameplay
 */
export const defaultCarConfig: CarConfig = {
  maxSpeed: 350,           // pixels per second
  maxReverseSpeed: 150,    // pixels per second
  acceleration: 250,       // pixels per second squared
  braking: 400,            // pixels per second squared (brakes are stronger)
  friction: 100,           // natural slowdown
  baseTurnRate: 3.5,       // radians per second at reference speed
  turnSpeedFactor: 0.8,    // turn rate scales with speed
  minSpeedToTurn: 5,       // minimum speed to allow turning
  collisionSpeedLoss: 0.3, // lose 30% speed on collision
  collisionPushStrength: 50, // push force back onto track
  collisionRadius: 20,     // hitbox radius
};

/**
 * Complete car state including physics configuration
 */
export interface CarState extends VehicleState {
  /** Whether the car is currently colliding with a wall */
  isColliding: boolean;
  /** Physics configuration */
  config: CarConfig;
}

/**
 * Create initial car state at a given position and angle
 */
export function createCarState(
  position: Point,
  angle: number,
  config: CarConfig = defaultCarConfig
): CarState {
  return {
    position: { ...position },
    angle,
    velocity: { x: 0, y: 0 },
    speed: 0,
    isColliding: false,
    config,
  };
}

/**
 * Update car physics based on input and deltaTime
 * This is the main physics update function called each frame
 *
 * @param car - Current car state (will be mutated)
 * @param input - Current input state
 * @param deltaTime - Time since last frame in seconds
 * @returns Updated car state (same reference, mutated)
 */
export function updateCar(
  car: CarState,
  input: InputState,
  deltaTime: number
): CarState {
  // Clamp deltaTime to prevent physics explosion on lag spikes
  const dt = Math.min(deltaTime, 0.05);

  // === STEERING ===
  updateSteering(car, input, dt);

  // === ACCELERATION / BRAKING ===
  updateSpeed(car, input, dt);

  // === APPLY VELOCITY ===
  updatePosition(car, dt);

  // === COLLISION DETECTION AND RESPONSE ===
  handleCollisions(car);

  return car;
}

/**
 * Update car steering based on input
 */
function updateSteering(car: CarState, input: InputState, dt: number): void {
  // Only allow turning if car is moving
  if (Math.abs(car.speed) < car.config.minSpeedToTurn) {
    return;
  }

  // Calculate turn rate based on speed
  // Turn rate decreases as speed increases for more realistic feel
  const speedRatio = Math.abs(car.speed) / car.config.maxSpeed;
  const turnRate = car.config.baseTurnRate * (1 - speedRatio * car.config.turnSpeedFactor);

  // Apply steering
  let steerDirection = 0;
  if (input.steerLeft) steerDirection -= 1;
  if (input.steerRight) steerDirection += 1;

  // Reverse steering direction when going backwards
  if (car.speed < 0) {
    steerDirection *= -1;
  }

  car.angle += steerDirection * turnRate * dt;

  // Normalize angle to -PI to PI range
  while (car.angle > Math.PI) car.angle -= Math.PI * 2;
  while (car.angle < -Math.PI) car.angle += Math.PI * 2;
}

/**
 * Update car speed based on input (acceleration, braking, friction)
 */
function updateSpeed(car: CarState, input: InputState, dt: number): void {
  const { config } = car;

  let speedChange = 0;

  if (input.accelerate) {
    // Accelerate forward
    speedChange += config.acceleration * dt;
  }

  if (input.brake) {
    if (car.speed > 0) {
      // Braking while moving forward
      speedChange -= config.braking * dt;
    } else {
      // Reversing when stopped or already moving backward
      speedChange -= config.acceleration * 0.6 * dt; // Reverse is slower
    }
  }

  // Apply natural friction/drag when not accelerating
  if (!input.accelerate && !input.brake) {
    if (car.speed > 0) {
      speedChange -= config.friction * dt;
      // Don't let friction make us go backwards
      if (car.speed + speedChange < 0) {
        speedChange = -car.speed;
      }
    } else if (car.speed < 0) {
      speedChange += config.friction * dt;
      // Don't let friction make us go forwards
      if (car.speed + speedChange > 0) {
        speedChange = -car.speed;
      }
    }
  }

  // Update speed
  car.speed += speedChange;

  // Clamp speed to limits
  car.speed = Math.max(-config.maxReverseSpeed, Math.min(config.maxSpeed, car.speed));

  // Update velocity vector based on speed and angle
  car.velocity.x = Math.cos(car.angle) * car.speed;
  car.velocity.y = Math.sin(car.angle) * car.speed;
}

/**
 * Update car position based on velocity
 */
function updatePosition(car: CarState, dt: number): void {
  car.position.x += car.velocity.x * dt;
  car.position.y += car.velocity.y * dt;
}

/**
 * Handle collision detection and response
 */
function handleCollisions(car: CarState): void {
  const { config } = car;

  // Check if car center is on track
  const centerOnTrack = isPointOnTrack(car.position);

  // Also check points around the car for more accurate collision
  const checkPoints = [
    { x: car.position.x + Math.cos(car.angle) * config.collisionRadius,
      y: car.position.y + Math.sin(car.angle) * config.collisionRadius },
    { x: car.position.x - Math.cos(car.angle) * config.collisionRadius,
      y: car.position.y - Math.sin(car.angle) * config.collisionRadius },
    { x: car.position.x + Math.cos(car.angle + Math.PI/2) * config.collisionRadius * 0.6,
      y: car.position.y + Math.sin(car.angle + Math.PI/2) * config.collisionRadius * 0.6 },
    { x: car.position.x + Math.cos(car.angle - Math.PI/2) * config.collisionRadius * 0.6,
      y: car.position.y + Math.sin(car.angle - Math.PI/2) * config.collisionRadius * 0.6 },
  ];

  const allPointsOnTrack = checkPoints.every(p => isPointOnTrack(p));

  if (!centerOnTrack || !allPointsOnTrack) {
    car.isColliding = true;

    // Get normal pointing toward track center
    const normal = getTrackNormal(car.position);

    // Push car back onto track
    car.position.x += normal.x * config.collisionPushStrength * 0.1;
    car.position.y += normal.y * config.collisionPushStrength * 0.1;

    // Reduce speed on collision
    car.speed *= (1 - config.collisionSpeedLoss);

    // Update velocity to match new speed
    car.velocity.x = Math.cos(car.angle) * car.speed;
    car.velocity.y = Math.sin(car.angle) * car.speed;

    // Additional push if still off track after initial correction
    let iterations = 0;
    while (!isPointOnTrack(car.position) && iterations < 10) {
      car.position.x += normal.x * config.collisionPushStrength * 0.2;
      car.position.y += normal.y * config.collisionPushStrength * 0.2;
      iterations++;
    }
  } else {
    car.isColliding = false;
  }
}

/**
 * Reset car to a specific position and angle
 * Useful for respawning or starting a new race
 */
export function resetCar(
  car: CarState,
  position: Point,
  angle: number
): CarState {
  car.position = { ...position };
  car.angle = angle;
  car.velocity = { x: 0, y: 0 };
  car.speed = 0;
  car.isColliding = false;
  return car;
}

/**
 * Get a read-only view of the vehicle state
 * Useful for rendering or other systems that shouldn't modify the car
 */
export function getVehicleState(car: CarState): VehicleState {
  return {
    position: { ...car.position },
    angle: car.angle,
    velocity: { ...car.velocity },
    speed: car.speed,
  };
}

/**
 * Create input state for AI/CPU control
 * Can be used by Story 5 for CPU cars
 */
export function createInputFromDirection(
  accelerate: boolean,
  brake: boolean,
  steerLeft: boolean,
  steerRight: boolean
): InputState {
  return { accelerate, brake, steerLeft, steerRight };
}
