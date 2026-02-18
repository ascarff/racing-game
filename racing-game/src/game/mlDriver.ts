/**
 * ML Driver System for the 2D racing game
 * Provides ML-based AI opponent using trained neural network
 *
 * Features:
 * - Load trained model weights from JSON
 * - Build observations matching the training environment
 * - Run inference to get actions
 * - Convert actions to InputState for car physics
 * - Fallback simple network if no trained model available
 */

import type { Point } from './types';
import type { InputState } from './input';
import {
  createCarState,
  updateCar,
  type CarState,
  type CarConfig,
  defaultCarConfig,
} from './car';
import { isPointOnTrack, trackData } from './track';

// Track constants (must match track.ts)
const CANVAS_WIDTH = 1200;
const CANVAS_HEIGHT = 800;
const CENTER_X = CANVAS_WIDTH / 2;
const CENTER_Y = CANVAS_HEIGHT / 2;

/**
 * Action mapping (matching Python environment)
 * Discrete(9) action space
 */
const ACTIONS: Record<number, InputState> = {
  0: { accelerate: false, brake: false, steerLeft: false, steerRight: false }, // Coast
  1: { accelerate: true, brake: false, steerLeft: false, steerRight: false },  // Accelerate
  2: { accelerate: false, brake: true, steerLeft: false, steerRight: false },  // Brake
  3: { accelerate: false, brake: false, steerLeft: true, steerRight: false },  // Steer left
  4: { accelerate: false, brake: false, steerLeft: false, steerRight: true },  // Steer right
  5: { accelerate: true, brake: false, steerLeft: true, steerRight: false },   // Accel + left
  6: { accelerate: true, brake: false, steerLeft: false, steerRight: true },   // Accel + right
  7: { accelerate: false, brake: true, steerLeft: true, steerRight: false },   // Brake + left
  8: { accelerate: false, brake: true, steerLeft: false, steerRight: true },   // Brake + right
};

/**
 * Convert action index to InputState
 */
export function actionToInput(action: number): InputState {
  return ACTIONS[action] || ACTIONS[0];
}

/**
 * Model layer definition (from exported JSON)
 */
interface LinearLayer {
  name: string;
  type: 'linear';
  weight: number[][];
  bias: number[];
  in_features: number;
  out_features: number;
}

/**
 * Model weights structure (from exported JSON)
 */
interface ModelWeights {
  meta: {
    obs_dim: number;
    n_actions: number;
    format: string;
  };
  layers: LinearLayer[];
  value_layers?: LinearLayer[];
}

/**
 * ML Driver state
 */
export interface MLDriver {
  /** The car physics state */
  carState: CarState;
  /** Whether the ML model is loaded */
  modelLoaded: boolean;
  /** Model weights (null if using fallback) */
  weights: ModelWeights | null;
  /** Last computed observation (for debugging) */
  lastObservation: number[] | null;
  /** Last selected action (for debugging) */
  lastAction: number;
  /** Checkpoints passed (for observation) */
  checkpointsPassed: Set<number>;
  /** Last position for tracking */
  lastPosition: Point;
  /** Using fallback simple model */
  usingFallback: boolean;
}

/**
 * Simple fallback neural network weights
 * Hand-tuned to provide reasonable racing behavior
 * Architecture: 15 inputs -> 32 hidden -> 9 outputs
 */
function createFallbackWeights(): ModelWeights {
  // Create a simple heuristic-based "network" that:
  // - Uses wall distances to steer away from walls
  // - Uses speed to decide acceleration/braking
  // - Uses heading to stay on track

  const obsSize = 15;
  const hiddenSize = 32;
  const outputSize = 9;

  // Initialize with small random weights for variety
  const seededRandom = (seed: number) => {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
  };

  // First layer: obs -> hidden
  const layer1Weight: number[][] = [];
  const layer1Bias: number[] = [];
  for (let i = 0; i < hiddenSize; i++) {
    const row: number[] = [];
    for (let j = 0; j < obsSize; j++) {
      // Hand-tune some connections for racing behavior
      let w = (seededRandom(i * obsSize + j) - 0.5) * 0.1;

      // Boost connections from wall distances (indices 8-12) to steering-related neurons
      if (j >= 8 && j <= 12) {
        if (i < 8) {
          // Left wall rays connect to "steer right" neurons
          if (j === 8 || j === 9) w = (seededRandom(i * obsSize + j) - 0.3) * 0.5;
          // Right wall rays connect to "steer left" neurons
          if (j === 11 || j === 12) w = (seededRandom(i * obsSize + j) - 0.7) * 0.5;
        }
      }

      // Speed (index 4) affects acceleration decisions
      if (j === 4 && i >= 8 && i < 16) {
        w = -0.3; // Low speed -> accelerate
      }

      row.push(w);
    }
    layer1Weight.push(row);
    layer1Bias.push((seededRandom(i + 1000) - 0.5) * 0.1);
  }

  // Output layer: hidden -> actions
  // Actions: 0=coast, 1=accel, 2=brake, 3=left, 4=right, 5=accel+left, 6=accel+right, 7=brake+left, 8=brake+right
  const layer2Weight: number[][] = [];
  const layer2Bias: number[] = [];
  for (let i = 0; i < outputSize; i++) {
    const row: number[] = [];
    for (let j = 0; j < hiddenSize; j++) {
      let w = (seededRandom(i * hiddenSize + j + 5000) - 0.5) * 0.1;

      // Bias toward acceleration actions (1, 5, 6)
      if (i === 1 || i === 5 || i === 6) {
        if (j >= 8 && j < 16) w += 0.2;
      }

      row.push(w);
    }
    layer2Weight.push(row);

    // Bias the outputs
    let bias = 0;
    if (i === 1) bias = 0.5;      // Prefer accelerate
    if (i === 5) bias = 0.3;      // Accel + left is good
    if (i === 6) bias = 0.3;      // Accel + right is good
    if (i === 0) bias = -0.5;     // Avoid coasting
    layer2Bias.push(bias);
  }

  return {
    meta: {
      obs_dim: obsSize,
      n_actions: outputSize,
      format: 'fallback_heuristic',
    },
    layers: [
      {
        name: 'policy_net.0',
        type: 'linear',
        weight: layer1Weight,
        bias: layer1Bias,
        in_features: obsSize,
        out_features: hiddenSize,
      },
      {
        name: 'action_net',
        type: 'linear',
        weight: layer2Weight,
        bias: layer2Bias,
        in_features: hiddenSize,
        out_features: outputSize,
      },
    ],
  };
}

/**
 * Apply a linear layer
 */
function linearLayer(input: number[], layer: LinearLayer): number[] {
  const output: number[] = new Array(layer.out_features).fill(0);
  for (let i = 0; i < layer.out_features; i++) {
    output[i] = layer.bias[i];
    for (let j = 0; j < layer.in_features; j++) {
      output[i] += layer.weight[i][j] * input[j];
    }
  }
  return output;
}

/**
 * Apply tanh activation
 */
function tanh(x: number[]): number[] {
  return x.map(v => Math.tanh(v));
}

/**
 * Apply softmax to get probabilities
 */
function softmax(x: number[]): number[] {
  const max = Math.max(...x);
  const exp = x.map(v => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(v => v / sum);
}

/**
 * Get action from model prediction
 */
function predict(weights: ModelWeights, observation: number[], deterministic: boolean = true): number {
  // Forward pass through policy network
  let x = observation;

  // Apply each layer with tanh activation (except last)
  for (let i = 0; i < weights.layers.length - 1; i++) {
    x = linearLayer(x, weights.layers[i]);
    x = tanh(x);
  }

  // Final layer (action logits)
  const logits = linearLayer(x, weights.layers[weights.layers.length - 1]);

  if (deterministic) {
    // Select action with highest logit
    return logits.indexOf(Math.max(...logits));
  } else {
    // Sample from probability distribution
    const probs = softmax(logits);
    const random = Math.random();
    let cumsum = 0;
    for (let i = 0; i < probs.length; i++) {
      cumsum += probs[i];
      if (random < cumsum) {
        return i;
      }
    }
    return probs.length - 1;
  }
}

/**
 * Cast a ray from a point and find distance to track boundary
 */
function castRay(x: number, y: number, angle: number, maxDistance: number = 300): number {
  const dx = Math.cos(angle);
  const dy = Math.sin(angle);
  const step = 3; // Step size for ray marching

  for (let distance = 0; distance <= maxDistance; distance += step) {
    const testX = x + dx * distance;
    const testY = y + dy * distance;

    if (!isPointOnTrack({ x: testX, y: testY })) {
      return distance;
    }
  }

  return maxDistance;
}

/**
 * Get distances to track boundaries using raycasting
 * Rays spread from -90 to +90 degrees relative to heading
 */
function getWallDistances(x: number, y: number, angle: number, numRays: number = 5): number[] {
  const distances: number[] = [];
  const maxDistance = 300;

  // Spread rays from -90 to +90 degrees relative to heading
  for (let i = 0; i < numRays; i++) {
    const rayOffset = -Math.PI / 2 + (i / (numRays - 1)) * Math.PI;
    const rayAngle = angle + rayOffset;
    distances.push(castRay(x, y, rayAngle, maxDistance));
  }

  return distances;
}

/**
 * Get track progress angle (0 to 2*PI, starting from bottom going counter-clockwise)
 */
function getTrackProgress(x: number, y: number): number {
  const dx = x - CENTER_X;
  const dy = y - CENTER_Y;

  // atan2 returns angle where 0 is right, PI/2 is down, PI/-PI is left, -PI/2 is up
  // We want 0 to be at the bottom (start/finish), increasing counter-clockwise
  const angle = Math.atan2(dy, dx);

  // Shift so bottom (PI/2) becomes 0, and goes counter-clockwise
  let progressAngle = Math.PI / 2 - angle;

  // Normalize to 0 to 2*PI
  if (progressAngle < 0) {
    progressAngle += 2 * Math.PI;
  }

  return progressAngle;
}

/**
 * Build observation vector from car state
 * Matches the Python training environment exactly
 *
 * Observation Space (15 features):
 * - [0]: x position (normalized to 0-1)
 * - [1]: y position (normalized to 0-1)
 * - [2]: cos(angle) - heading x component
 * - [3]: sin(angle) - heading y component
 * - [4]: speed (normalized to -1 to 1)
 * - [5]: velocity_x (normalized)
 * - [6]: velocity_y (normalized)
 * - [7]: is_colliding (0 or 1)
 * - [8-12]: distance to track boundaries (5 rays, normalized)
 * - [13]: progress around track (0 to 1)
 * - [14]: checkpoint progress (0 to 1)
 */
function buildObservation(driver: MLDriver): number[] {
  const car = driver.carState;
  const maxSpeed = car.config.maxSpeed;
  const maxDistance = 300;
  const numCheckpoints = trackData.checkpoints.length;

  // Normalize values
  const normX = car.position.x / CANVAS_WIDTH;
  const normY = car.position.y / CANVAS_HEIGHT;
  const cosAngle = Math.cos(car.angle);
  const sinAngle = Math.sin(car.angle);
  const normSpeed = car.speed / maxSpeed;
  const normVx = car.velocity.x / maxSpeed;
  const normVy = car.velocity.y / maxSpeed;
  const isColliding = car.isColliding ? 1.0 : 0.0;

  // Get wall distances (5 rays)
  const wallDistances = getWallDistances(car.position.x, car.position.y, car.angle, 5);
  const normDistances = wallDistances.map(d => d / maxDistance);

  // Progress features
  const trackProgress = getTrackProgress(car.position.x, car.position.y) / (2 * Math.PI);
  const checkpointProgress = driver.checkpointsPassed.size / (numCheckpoints + 1);

  // Build observation array
  const observation = [
    normX,
    normY,
    cosAngle,
    sinAngle,
    normSpeed,
    normVx,
    normVy,
    isColliding,
    ...normDistances,
    trackProgress,
    checkpointProgress,
  ];

  // Clip to [-1, 1] range
  return observation.map(v => Math.max(-1, Math.min(1, v)));
}

/**
 * Check if a line segment crosses a checkpoint
 */
function lineSegmentsCross(
  p1x: number, p1y: number, p2x: number, p2y: number,
  p3x: number, p3y: number, p4x: number, p4y: number
): boolean {
  const ccw = (ax: number, ay: number, bx: number, by: number, cx: number, cy: number): boolean => {
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax);
  };

  return (ccw(p1x, p1y, p3x, p3y, p4x, p4y) !== ccw(p2x, p2y, p3x, p3y, p4x, p4y)) &&
         (ccw(p1x, p1y, p2x, p2y, p3x, p3y) !== ccw(p1x, p1y, p2x, p2y, p4x, p4y));
}

/**
 * Update checkpoint tracking
 */
function updateCheckpoints(driver: MLDriver): void {
  const oldX = driver.lastPosition.x;
  const oldY = driver.lastPosition.y;
  const newX = driver.carState.position.x;
  const newY = driver.carState.position.y;

  for (const checkpoint of trackData.checkpoints) {
    if (!driver.checkpointsPassed.has(checkpoint.id)) {
      const crossed = lineSegmentsCross(
        oldX, oldY, newX, newY,
        checkpoint.line.start.x, checkpoint.line.start.y,
        checkpoint.line.end.x, checkpoint.line.end.y
      );
      if (crossed) {
        driver.checkpointsPassed.add(checkpoint.id);
      }
    }
  }

  // Check for lap completion (crossing start/finish with all checkpoints)
  if (driver.checkpointsPassed.size >= trackData.checkpoints.length) {
    const sf = trackData.startFinishLine;
    const crossed = lineSegmentsCross(
      oldX, oldY, newX, newY,
      sf.start.x, sf.start.y,
      sf.end.x, sf.end.y
    );
    if (crossed) {
      // Reset checkpoints for next lap
      driver.checkpointsPassed.clear();
    }
  }
}

/**
 * Attempt to load model weights from a URL
 */
export async function loadMLModel(weightsUrl: string): Promise<ModelWeights | null> {
  try {
    const response = await fetch(weightsUrl);
    if (!response.ok) {
      console.warn(`Failed to load ML model from ${weightsUrl}: ${response.status}`);
      return null;
    }
    const weights = await response.json() as ModelWeights;
    console.log(`ML model loaded: ${weights.meta.obs_dim} inputs, ${weights.meta.n_actions} actions`);
    return weights;
  } catch (error) {
    console.warn('Failed to load ML model:', error);
    return null;
  }
}

/**
 * Create an ML driver at a given position
 * Optionally provide pre-loaded weights, otherwise uses fallback
 */
export function createMLDriver(
  position: Point,
  angle: number,
  weights: ModelWeights | null = null,
  carConfig?: Partial<CarConfig>
): MLDriver {
  // Create ML car config - can be same as player for fair competition
  const mlCarConfig: CarConfig = {
    ...defaultCarConfig,
    maxSpeed: 330, // Slightly slower to compensate for potentially better pathing
    acceleration: 240,
    ...carConfig,
  };

  const usingFallback = weights === null;
  const modelWeights = weights || createFallbackWeights();

  return {
    carState: createCarState(position, angle, mlCarConfig),
    modelLoaded: weights !== null,
    weights: modelWeights,
    lastObservation: null,
    lastAction: 1, // Default to accelerate
    checkpointsPassed: new Set(),
    lastPosition: { ...position },
    usingFallback,
  };
}

/**
 * Update the ML driver for one frame
 * Runs inference and updates car physics
 */
export function updateMLDriver(driver: MLDriver, deltaTime: number): void {
  // Store last position for checkpoint tracking
  driver.lastPosition = { ...driver.carState.position };

  // Build observation from current state
  const observation = buildObservation(driver);
  driver.lastObservation = observation;

  // Run inference to get action
  if (driver.weights) {
    // Use deterministic=true for consistent behavior during gameplay
    // Can use deterministic=false for more varied behavior
    driver.lastAction = predict(driver.weights, observation, true);
  }

  // Convert action to input
  const input = actionToInput(driver.lastAction);

  // Update car physics
  updateCar(driver.carState, input, deltaTime);

  // Update checkpoint tracking
  updateCheckpoints(driver);
}

/**
 * Get ML car state for rendering and collision detection
 */
export function getMLCarState(driver: MLDriver): CarState {
  return driver.carState;
}

/**
 * Reset ML driver to starting position
 */
export function resetMLDriver(driver: MLDriver, position: Point, angle: number): void {
  driver.carState.position = { ...position };
  driver.carState.angle = angle;
  driver.carState.velocity = { x: 0, y: 0 };
  driver.carState.speed = 0;
  driver.carState.isColliding = false;
  driver.lastPosition = { ...position };
  driver.lastObservation = null;
  driver.lastAction = 1;
  driver.checkpointsPassed.clear();
}

/**
 * Check if the ML driver is using the fallback model
 */
export function isUsingFallback(driver: MLDriver): boolean {
  return driver.usingFallback;
}

/**
 * Get debug info for the ML driver
 */
export function getMLDriverDebugInfo(driver: MLDriver): {
  modelLoaded: boolean;
  usingFallback: boolean;
  lastAction: number;
  actionName: string;
  checkpointsPassed: number;
  observation: number[] | null;
} {
  const actionNames = [
    'Coast', 'Accelerate', 'Brake', 'Left', 'Right',
    'Accel+Left', 'Accel+Right', 'Brake+Left', 'Brake+Right'
  ];

  return {
    modelLoaded: driver.modelLoaded,
    usingFallback: driver.usingFallback,
    lastAction: driver.lastAction,
    actionName: actionNames[driver.lastAction] || 'Unknown',
    checkpointsPassed: driver.checkpointsPassed.size,
    observation: driver.lastObservation,
  };
}

/**
 * Draw ML driver debug visualization
 */
export function drawMLDriverDebug(
  ctx: CanvasRenderingContext2D,
  driver: MLDriver
): void {
  const car = driver.carState;

  ctx.save();

  // Draw wall distance rays
  const wallDistances = getWallDistances(car.position.x, car.position.y, car.angle, 5);
  const rayAngles = [-Math.PI / 2, -Math.PI / 4, 0, Math.PI / 4, Math.PI / 2];

  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i++) {
    const rayAngle = car.angle + rayAngles[i];
    const distance = wallDistances[i];

    const endX = car.position.x + Math.cos(rayAngle) * distance;
    const endY = car.position.y + Math.sin(rayAngle) * distance;

    // Color based on distance (red = close, green = far)
    const ratio = distance / 300;
    ctx.strokeStyle = `rgb(${Math.floor(255 * (1 - ratio))}, ${Math.floor(255 * ratio)}, 0)`;
    ctx.beginPath();
    ctx.moveTo(car.position.x, car.position.y);
    ctx.lineTo(endX, endY);
    ctx.stroke();

    // Draw endpoint
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.arc(endX, endY, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  // Draw action indicator
  const actionNames = ['C', 'A', 'B', 'L', 'R', 'AL', 'AR', 'BL', 'BR'];
  ctx.fillStyle = '#ffffff';
  ctx.font = '12px monospace';
  ctx.fillText(
    `ML: ${actionNames[driver.lastAction]}`,
    car.position.x + 25,
    car.position.y - 25
  );

  ctx.restore();
}
