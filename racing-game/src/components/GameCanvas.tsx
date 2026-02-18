/**
 * Main game canvas component
 * Sets up the canvas and game loop using requestAnimationFrame
 */

import { useRef, useEffect, useCallback, useState } from 'react';
import { trackData, defaultTrackRenderConfig } from '../game/track';
import { drawTrack, drawVehicle } from '../game/renderer';
import { initInput, cleanupInput, getInputState } from '../game/input';
import { createCarState, updateCar, type CarState } from '../game/car';
import { formatTime } from '../game/gameState';
import {
  createLapTimer,
  startLapTimer,
  updateLapTimer,
  pauseLapTimer,
  resumeLapTimer,
  getBestLapTime,
  type LapTimerState,
} from '../game/lapTimer';
import {
  createCPUDriver,
  updateCPUDriver,
  drawWaypoints,
  type CPUDriver,
} from '../game/cpuDriver';
import {
  createMLDriver,
  updateMLDriver,
  loadMLModel,
  getMLDriverDebugInfo,
  drawMLDriverDebug,
  type MLDriver,
} from '../game/mlDriver';
import { Countdown } from './Countdown';
import type { GameState, Point } from '../game/types';

/**
 * Race finish data passed to onRaceFinished callback
 */
export interface RaceFinishData {
  /** Player's lap times */
  playerLapTimes: number[];
  /** Player's total time */
  playerTotalTime: number;
  /** CPU's lap times */
  cpuLapTimes: number[];
  /** CPU's total time */
  cpuTotalTime: number;
  /** Player's position (1 = won, 2 = lost) */
  playerPosition: 1 | 2;
}

interface GameCanvasProps {
  /** Show debug information like checkpoints */
  showDebug?: boolean;
  /** Total number of laps for the race (0 = unlimited/practice mode) */
  totalLaps?: number;
  /** Whether the game is paused */
  isPaused?: boolean;
  /** Callback when race is finished */
  onRaceFinished?: (data: RaceFinishData) => void;
  /** Best lap time to beat (from localStorage) */
  bestLapTime?: number | null;
  /** Whether a new lap record was just set */
  newLapRecord?: boolean;
  /** Use ML-based CPU driver instead of waypoint AI */
  useMLDriver?: boolean;
  /** URL to model weights JSON (optional, uses fallback if not provided) */
  modelWeightsUrl?: string;
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
 * Handle collision between two cars
 * Returns true if cars are colliding and modifies their states
 */
function handleCarCollision(car1: CarState, car2: CarState): boolean {
  const collisionRadius = car1.config.collisionRadius + car2.config.collisionRadius;
  const dist = distance(car1.position, car2.position);

  if (dist < collisionRadius && dist > 0) {
    // Calculate collision normal (direction from car1 to car2)
    const nx = (car2.position.x - car1.position.x) / dist;
    const ny = (car2.position.y - car1.position.y) / dist;

    // Calculate overlap
    const overlap = collisionRadius - dist;

    // Separate cars (push them apart equally)
    const pushAmount = overlap / 2 + 2; // Extra 2px to prevent sticking

    car1.position.x -= nx * pushAmount;
    car1.position.y -= ny * pushAmount;
    car2.position.x += nx * pushAmount;
    car2.position.y += ny * pushAmount;

    // Calculate relative velocity along collision normal
    const relVelX = car1.velocity.x - car2.velocity.x;
    const relVelY = car1.velocity.y - car2.velocity.y;
    const relVelNormal = relVelX * nx + relVelY * ny;

    // Only resolve if cars are moving toward each other
    if (relVelNormal > 0) {
      // Simple elastic collision with some energy loss
      const restitution = 0.6; // Bounciness (0 = no bounce, 1 = perfect bounce)
      const impulse = relVelNormal * restitution;

      // Apply impulse to velocities
      car1.velocity.x -= impulse * nx;
      car1.velocity.y -= impulse * ny;
      car2.velocity.x += impulse * nx;
      car2.velocity.y += impulse * ny;

      // Update speeds to match new velocities
      car1.speed = Math.sqrt(car1.velocity.x ** 2 + car1.velocity.y ** 2);
      car2.speed = Math.sqrt(car2.velocity.x ** 2 + car2.velocity.y ** 2);

      // Reduce speeds slightly on collision
      const speedLoss = 0.85;
      car1.speed *= speedLoss;
      car2.speed *= speedLoss;
      car1.velocity.x *= speedLoss;
      car1.velocity.y *= speedLoss;
      car2.velocity.x *= speedLoss;
      car2.velocity.y *= speedLoss;
    }

    return true;
  }

  return false;
}

/**
 * Calculate CPU car start position (offset from player)
 */
function getCPUStartPosition(): Point {
  // Position CPU car slightly behind and to the side of player
  const offset = 45; // Offset to the outside of the track
  return {
    x: trackData.startPosition.x + offset,
    y: trackData.startPosition.y + 60, // Slightly behind
  };
}

/**
 * GameCanvas component - renders the racing game
 */
export function GameCanvas({
  showDebug = false,
  totalLaps = 3,
  isPaused = false,
  onRaceFinished,
  bestLapTime = null,
  newLapRecord = false,
  useMLDriver = true,
  modelWeightsUrl,
}: GameCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const gameStateRef = useRef<GameState>({
    timestamp: 0,
    deltaTime: 0,
    isRunning: true,
  });
  const animationFrameRef = useRef<number>(0);

  // Player car state
  const playerCarRef = useRef<CarState>(
    createCarState(trackData.startPosition, trackData.startAngle)
  );

  // CPU car state (waypoint-based AI)
  const cpuDriverRef = useRef<CPUDriver>(
    createCPUDriver(getCPUStartPosition(), trackData.startAngle)
  );

  // ML driver state (ML-based AI)
  const mlDriverRef = useRef<MLDriver | null>(null);

  // Track which AI mode is active
  const [aiMode, setAiMode] = useState<'ml' | 'waypoint' | 'loading'>('loading');

  // Track previous car positions for line crossing detection
  const lastPlayerPositionRef = useRef<Point>({ ...trackData.startPosition });

  // Lap timer state for player - using useState for UI updates
  const [playerLapTimerState, setPlayerLapTimerState] = useState<LapTimerState>(() =>
    createLapTimer({
      totalLaps,
      checkpointIds: trackData.checkpoints.map(cp => cp.id),
    })
  );

  // Lap timer state for CPU
  const [cpuLapTimerState, setCPULapTimerState] = useState<LapTimerState>(() =>
    createLapTimer({
      totalLaps,
      checkpointIds: trackData.checkpoints.map(cp => cp.id),
    })
  );

  // Track whether race has started (player has moved)
  const raceStartedRef = useRef<boolean>(false);

  // Track if cars are currently colliding (for visual feedback)
  const [carsColliding, setCarsColliding] = useState(false);

  // Countdown state - starts active
  const [isCountdownActive, setIsCountdownActive] = useState(true);
  const [isCountdownComplete, setIsCountdownComplete] = useState(false);

  // Ref to track pause state for use in game loop
  const isPausedRef = useRef<boolean>(isPaused);
  isPausedRef.current = isPaused;

  // Ref to track countdown complete state for game loop
  const isCountdownCompleteRef = useRef<boolean>(isCountdownComplete);
  isCountdownCompleteRef.current = isCountdownComplete;

  // Refs to track lap timer states for use in game loop
  const playerLapTimerRef = useRef<LapTimerState>(playerLapTimerState);
  playerLapTimerRef.current = playerLapTimerState;

  const cpuLapTimerRef = useRef<LapTimerState>(cpuLapTimerState);
  cpuLapTimerRef.current = cpuLapTimerState;

  /**
   * Handle countdown completion - start the race
   */
  const handleCountdownComplete = useCallback(() => {
    setIsCountdownComplete(true);
    // Start timers immediately when countdown completes
    raceStartedRef.current = true;
    setPlayerLapTimerState(prev => startLapTimer(prev));
    setCPULapTimerState(prev => startLapTimer(prev));
  }, []);

  /**
   * Update game logic
   */
  const update = useCallback((deltaTime: number) => {
    // Skip updates if paused or countdown not complete
    if (isPausedRef.current || !isCountdownCompleteRef.current) {
      return;
    }

    // Get current input state
    const input = getInputState();

    // Store last positions before updating cars
    const lastPlayerPosition = { ...playerCarRef.current.position };
    const lastCPUPosition = mlDriverRef.current
      ? { ...mlDriverRef.current.carState.position }
      : { ...cpuDriverRef.current.carState.position };

    // Update player car physics
    updateCar(playerCarRef.current, input, deltaTime);

    // Update CPU car - use ML driver if available, fallback to waypoint AI
    if (mlDriverRef.current) {
      updateMLDriver(mlDriverRef.current, deltaTime);
    } else {
      updateCPUDriver(cpuDriverRef.current, deltaTime);
    }

    // Handle car-to-car collision
    const cpuCarState = mlDriverRef.current
      ? mlDriverRef.current.carState
      : cpuDriverRef.current.carState;
    const colliding = handleCarCollision(playerCarRef.current, cpuCarState);
    setCarsColliding(colliding);

    // Update player lap timer
    if (raceStartedRef.current && playerLapTimerRef.current.isRunning) {
      const result = updateLapTimer(
        playerLapTimerRef.current,
        playerCarRef.current.position,
        lastPlayerPosition,
        deltaTime,
        trackData.checkpoints,
        trackData.startFinishLine
      );

      // Only update state if something changed
      if (result.lapCompleted || result.checkpointCrossed !== null ||
          result.state.currentLapTime !== playerLapTimerRef.current.currentLapTime) {
        setPlayerLapTimerState(result.state);

        // Check if race finished
        if (result.state.raceFinished && onRaceFinished) {
          // Calculate final times and positions
          const playerTotalTime = result.state.lapTimes.reduce((sum, t) => sum + t, 0);
          const cpuTotalTime = cpuLapTimerRef.current.lapTimes.reduce((sum, t) => sum + t, 0);

          // Player wins if they have lower total time, or if CPU hasn't finished yet
          const playerPosition: 1 | 2 =
            cpuLapTimerRef.current.raceFinished
              ? (playerTotalTime <= cpuTotalTime ? 1 : 2)
              : 1; // Player finished first, so they win

          onRaceFinished({
            playerLapTimes: result.state.lapTimes,
            playerTotalTime,
            cpuLapTimes: cpuLapTimerRef.current.lapTimes,
            cpuTotalTime: cpuLapTimerRef.current.raceFinished ? cpuTotalTime : cpuTotalTime + cpuLapTimerRef.current.currentLapTime,
            playerPosition,
          });
        }
      }
    }

    // Update CPU lap timer
    if (raceStartedRef.current && cpuLapTimerRef.current.isRunning) {
      const cpuPosition = mlDriverRef.current
        ? mlDriverRef.current.carState.position
        : cpuDriverRef.current.carState.position;
      const result = updateLapTimer(
        cpuLapTimerRef.current,
        cpuPosition,
        lastCPUPosition,
        deltaTime,
        trackData.checkpoints,
        trackData.startFinishLine
      );

      // Only update state if something changed
      if (result.lapCompleted || result.checkpointCrossed !== null ||
          result.state.currentLapTime !== cpuLapTimerRef.current.currentLapTime) {
        setCPULapTimerState(result.state);

        // Check if CPU finished first (player lost)
        if (result.state.raceFinished && !playerLapTimerRef.current.raceFinished && onRaceFinished) {
          const playerTotalTime = playerLapTimerRef.current.lapTimes.reduce((sum, t) => sum + t, 0) + playerLapTimerRef.current.currentLapTime;
          const cpuTotalTime = result.state.lapTimes.reduce((sum, t) => sum + t, 0);

          onRaceFinished({
            playerLapTimes: playerLapTimerRef.current.lapTimes,
            playerTotalTime,
            cpuLapTimes: result.state.lapTimes,
            cpuTotalTime,
            playerPosition: 2, // CPU finished first, player loses
          });
        }
      }
    }

    // Update last position reference
    lastPlayerPositionRef.current = lastPlayerPosition;
  }, [onRaceFinished]);

  /**
   * Main render function - draws the current game state
   */
  const render = useCallback((ctx: CanvasRenderingContext2D) => {
    const playerCar = playerCarRef.current;
    const cpuCar = mlDriverRef.current
      ? mlDriverRef.current.carState
      : cpuDriverRef.current.carState;

    // Draw the track
    drawTrack(ctx, trackData, defaultTrackRenderConfig, showDebug);

    // Draw debug visualizations
    if (showDebug) {
      if (mlDriverRef.current) {
        // Draw ML driver debug (ray casts, action indicator)
        drawMLDriverDebug(ctx, mlDriverRef.current);
      } else {
        // Draw CPU waypoints for waypoint AI
        drawWaypoints(ctx, cpuDriverRef.current);
      }
    }

    // Draw the CPU car (blue for waypoint AI, purple for ML)
    const cpuBaseColor = mlDriverRef.current ? '#9b59b6' : '#3498db';
    const cpuCollideColor = mlDriverRef.current ? '#c39bd3' : '#6b9bff';
    const cpuColor = cpuCar.isColliding ? cpuCollideColor : cpuBaseColor;
    drawVehicle(ctx, cpuCar.position, cpuCar.angle, cpuColor);

    // Draw the player car (red) - draw on top so player can see their car
    const playerColor = playerCar.isColliding ? '#ff6b6b' : '#e74c3c';
    drawVehicle(ctx, playerCar.position, playerCar.angle, playerColor);

    // Draw collision indicator if cars are touching
    if (carsColliding) {
      const midX = (playerCar.position.x + cpuCar.position.x) / 2;
      const midY = (playerCar.position.y + cpuCar.position.y) / 2;

      ctx.save();
      ctx.fillStyle = '#ffff00';
      ctx.beginPath();
      ctx.arc(midX, midY, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }, [showDebug, carsColliding, aiMode]);

  /**
   * Game loop - called every frame via requestAnimationFrame
   */
  const gameLoop = useCallback((timestamp: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const state = gameStateRef.current;

    // Calculate delta time
    const deltaTime = state.timestamp > 0
      ? (timestamp - state.timestamp) / 1000
      : 0;

    // Update game state
    state.deltaTime = deltaTime;
    state.timestamp = timestamp;

    // Update game logic
    update(deltaTime);

    // Render the frame
    render(ctx);

    // Schedule next frame if game is running
    if (state.isRunning) {
      animationFrameRef.current = requestAnimationFrame(gameLoop);
    }
  }, [render, update]);

  /**
   * Handle pause state changes
   */
  useEffect(() => {
    if (raceStartedRef.current && playerLapTimerRef.current.isRunning !== !isPaused) {
      if (isPaused) {
        setPlayerLapTimerState(prev => pauseLapTimer(prev));
        setCPULapTimerState(prev => pauseLapTimer(prev));
      } else {
        setPlayerLapTimerState(prev => resumeLapTimer(prev));
        setCPULapTimerState(prev => resumeLapTimer(prev));
      }
    }
  }, [isPaused]);

  /**
   * Initialize the game loop and input system
   */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Initialize input handling
    initInput();

    // Reset player car to starting position
    playerCarRef.current = createCarState(
      trackData.startPosition,
      trackData.startAngle
    );

    // Reset CPU car to starting position
    const cpuStartPos = getCPUStartPosition();
    cpuDriverRef.current = createCPUDriver(cpuStartPos, trackData.startAngle);

    // Reset lap timers and countdown
    raceStartedRef.current = false;
    lastPlayerPositionRef.current = { ...trackData.startPosition };
    setIsCountdownActive(true);
    setIsCountdownComplete(false);

    setPlayerLapTimerState(createLapTimer({
      totalLaps,
      checkpointIds: trackData.checkpoints.map(cp => cp.id),
    }));

    setCPULapTimerState(createLapTimer({
      totalLaps,
      checkpointIds: trackData.checkpoints.map(cp => cp.id),
    }));

    // Initialize ML driver if enabled
    const initMLDriver = async () => {
      if (useMLDriver) {
        setAiMode('loading');
        let weights = null;

        // Try to load model weights if URL provided
        if (modelWeightsUrl) {
          weights = await loadMLModel(modelWeightsUrl);
        }

        // Create ML driver (with weights or fallback)
        mlDriverRef.current = createMLDriver(cpuStartPos, trackData.startAngle, weights);
        setAiMode('ml');

        console.log(
          weights
            ? 'ML Driver initialized with trained model'
            : 'ML Driver initialized with fallback heuristic model'
        );
      } else {
        // Use waypoint AI
        mlDriverRef.current = null;
        setAiMode('waypoint');
        console.log('Using waypoint-based CPU AI');
      }
    };

    initMLDriver();

    // Start the game loop
    gameStateRef.current.isRunning = true;
    animationFrameRef.current = requestAnimationFrame(gameLoop);

    // Cleanup on unmount
    return () => {
      gameStateRef.current.isRunning = false;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      cleanupInput();
    };
  }, [gameLoop, totalLaps, useMLDriver, modelWeightsUrl]);

  const playerCar = playerCarRef.current;
  const cpuCar = mlDriverRef.current
    ? mlDriverRef.current.carState
    : cpuDriverRef.current.carState;
  const playerBestLap = getBestLapTime(playerLapTimerState);
  const cpuBestLap = getBestLapTime(cpuLapTimerState);
  const mlDebugInfo = mlDriverRef.current ? getMLDriverDebugInfo(mlDriverRef.current) : null;

  // Calculate checkpoints passed count for player
  let playerCheckpointsPassedCount = 0;
  playerLapTimerState.checkpointsPassed.forEach((passed) => {
    if (passed) playerCheckpointsPassedCount++;
  });

  // Calculate checkpoints passed count for CPU
  let cpuCheckpointsPassedCount = 0;
  cpuLapTimerState.checkpointsPassed.forEach((passed) => {
    if (passed) cpuCheckpointsPassedCount++;
  });

  // Determine race position based on lap and checkpoint progress
  const playerProgress = (playerLapTimerState.currentLap - 1) * 100 + playerCheckpointsPassedCount * 25;
  const cpuProgress = (cpuLapTimerState.currentLap - 1) * 100 + cpuCheckpointsPassedCount * 25;
  const playerPosition = playerProgress >= cpuProgress ? 1 : 2;

  return (
    <div className="game-container" style={{ position: 'relative' }}>
      <canvas
        ref={canvasRef}
        width={trackData.dimensions.width}
        height={trackData.dimensions.height}
        style={{
          display: 'block',
          maxWidth: '100%',
          height: 'auto',
          border: '2px solid #333',
          borderRadius: '4px',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
        }}
      />

      {/* Countdown Overlay */}
      <Countdown
        isActive={isCountdownActive}
        onComplete={handleCountdownComplete}
      />

      {/* Race Position Indicator - only show after countdown */}
      {isCountdownComplete && (
      <div
        className="position-overlay"
        style={{
          position: 'absolute',
          top: '10px',
          left: '50%',
          transform: 'translateX(-50%)',
          padding: '8px 24px',
          backgroundColor: playerPosition === 1 ? 'rgba(46, 204, 113, 0.9)' : 'rgba(231, 76, 60, 0.9)',
          color: '#fff',
          fontFamily: 'monospace',
          fontSize: '24px',
          fontWeight: 'bold',
          borderRadius: '8px',
          textAlign: 'center',
        }}
      >
        {playerPosition === 1 ? '1ST' : '2ND'}
      </div>
      )}

      {/* Player Lap Timer UI Overlay (Left) */}
      <div
        className="lap-timer-overlay"
        style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          padding: '12px 16px',
          backgroundColor: 'rgba(231, 76, 60, 0.85)',
          color: '#fff',
          fontFamily: 'monospace',
          fontSize: '14px',
          borderRadius: '8px',
          minWidth: '160px',
          border: '2px solid #e74c3c',
        }}
      >
        <div style={{ fontWeight: 'bold', marginBottom: '6px', fontSize: '12px' }}>
          PLAYER (YOU)
        </div>
        {/* Current Lap Time */}
        <div style={{ marginBottom: '6px' }}>
          <span style={{ color: '#ffcccc', fontSize: '10px' }}>TIME</span>
          <div style={{ fontSize: '20px', fontWeight: 'bold', color: isPaused ? '#ffaa00' : '#fff' }}>
            {formatTime(playerLapTimerState.currentLapTime)}
          </div>
        </div>

        {/* Lap Counter */}
        <div style={{ marginBottom: '6px' }}>
          <span style={{ color: '#ffcccc', fontSize: '10px' }}>LAP</span>
          <div style={{ fontSize: '16px', fontWeight: 'bold' }}>
            {playerLapTimerState.totalLaps > 0
              ? `${Math.min(playerLapTimerState.currentLap, playerLapTimerState.totalLaps)} / ${playerLapTimerState.totalLaps}`
              : playerLapTimerState.currentLap}
          </div>
        </div>

        {/* Best Lap Time (if any laps completed) */}
        {playerBestLap !== null && (
          <div>
            <span style={{ color: '#ffcccc', fontSize: '10px' }}>BEST</span>
            <div style={{ fontSize: '14px', color: '#ffff00' }}>
              {formatTime(playerBestLap)}
            </div>
          </div>
        )}

        {/* Target time to beat (from localStorage) */}
        {bestLapTime !== null && (
          <div style={{ marginTop: '6px', borderTop: '1px solid rgba(255,255,255,0.2)', paddingTop: '6px' }}>
            <span style={{ color: '#ffcccc', fontSize: '10px' }}>TARGET</span>
            <div style={{ fontSize: '14px', color: '#00d4ff' }}>
              {formatTime(bestLapTime)}
            </div>
          </div>
        )}
      </div>

      {/* CPU Lap Timer UI Overlay (Right) */}
      <div
        className="cpu-timer-overlay"
        style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          padding: '12px 16px',
          backgroundColor: mlDriverRef.current ? 'rgba(155, 89, 182, 0.85)' : 'rgba(52, 152, 219, 0.85)',
          color: '#fff',
          fontFamily: 'monospace',
          fontSize: '14px',
          borderRadius: '8px',
          minWidth: '160px',
          border: mlDriverRef.current ? '2px solid #9b59b6' : '2px solid #3498db',
        }}
      >
        <div style={{ fontWeight: 'bold', marginBottom: '6px', fontSize: '12px' }}>
          {mlDriverRef.current
            ? (mlDebugInfo?.usingFallback ? 'CPU (ML Fallback)' : 'CPU (ML Trained)')
            : 'CPU (Waypoint AI)'}
        </div>
        {/* Current Lap Time */}
        <div style={{ marginBottom: '6px' }}>
          <span style={{ color: '#cce5ff', fontSize: '10px' }}>TIME</span>
          <div style={{ fontSize: '20px', fontWeight: 'bold', color: isPaused ? '#ffaa00' : '#fff' }}>
            {formatTime(cpuLapTimerState.currentLapTime)}
          </div>
        </div>

        {/* Lap Counter */}
        <div style={{ marginBottom: '6px' }}>
          <span style={{ color: '#cce5ff', fontSize: '10px' }}>LAP</span>
          <div style={{ fontSize: '16px', fontWeight: 'bold' }}>
            {cpuLapTimerState.totalLaps > 0
              ? `${Math.min(cpuLapTimerState.currentLap, cpuLapTimerState.totalLaps)} / ${cpuLapTimerState.totalLaps}`
              : cpuLapTimerState.currentLap}
          </div>
        </div>

        {/* Best Lap Time (if any laps completed) */}
        {cpuBestLap !== null && (
          <div>
            <span style={{ color: '#cce5ff', fontSize: '10px' }}>BEST</span>
            <div style={{ fontSize: '14px', color: '#ffff00' }}>
              {formatTime(cpuBestLap)}
            </div>
          </div>
        )}
      </div>

      {/* Race Status Overlays */}
      {playerLapTimerState.raceFinished && (
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            padding: '24px 48px',
            backgroundColor: 'rgba(0, 0, 0, 0.9)',
            color: '#fff',
            fontFamily: 'monospace',
            fontSize: '32px',
            fontWeight: 'bold',
            borderRadius: '16px',
            textAlign: 'center',
            border: playerPosition === 1 ? '4px solid #2ecc71' : '4px solid #e74c3c',
          }}
        >
          <div style={{ color: playerPosition === 1 ? '#2ecc71' : '#e74c3c' }}>
            {playerPosition === 1 ? 'YOU WIN!' : 'YOU LOSE'}
          </div>
          <div style={{ fontSize: '16px', marginTop: '12px', color: '#aaa' }}>
            Best Lap: {formatTime(playerBestLap)}
          </div>
          {/* New Record Notification */}
          {newLapRecord && (
            <div
              style={{
                marginTop: '16px',
                padding: '8px 16px',
                backgroundColor: 'rgba(255, 215, 0, 0.2)',
                border: '2px solid #ffd700',
                borderRadius: '8px',
                animation: 'pulse 1s ease-in-out infinite',
              }}
            >
              <div style={{ color: '#ffd700', fontSize: '18px', fontWeight: 'bold' }}>
                NEW RECORD!
              </div>
              <div style={{ color: '#ffd700', fontSize: '14px' }}>
                {formatTime(playerBestLap)}
              </div>
            </div>
          )}
        </div>
      )}

      {isPaused && !playerLapTimerState.raceFinished && (
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            padding: '24px 48px',
            backgroundColor: 'rgba(0, 0, 0, 0.85)',
            color: '#ffaa00',
            fontFamily: 'monospace',
            fontSize: '32px',
            fontWeight: 'bold',
            borderRadius: '16px',
            textAlign: 'center',
          }}
        >
          PAUSED
        </div>
      )}

      {/* Checkpoint Progress (debug mode only) */}
      {showDebug && (
        <div
          className="checkpoint-overlay"
          style={{
            position: 'absolute',
            bottom: '80px',
            right: '10px',
            padding: '12px 16px',
            backgroundColor: 'rgba(0, 0, 0, 0.75)',
            color: '#fff',
            fontFamily: 'monospace',
            fontSize: '12px',
            borderRadius: '8px',
          }}
        >
          <div style={{ display: 'flex', gap: '20px' }}>
            <div>
              <span style={{ color: '#e74c3c', fontSize: '10px' }}>PLAYER CPs</span>
              <div style={{ marginTop: '4px' }}>
                {Array.from(playerLapTimerState.checkpointsPassed.entries()).map(([id, passed]) => (
                  <span key={id} style={{ color: passed ? '#00ff00' : '#666', marginRight: '8px' }}>
                    {id}:{passed ? 'Y' : 'N'}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <span style={{ color: '#3498db', fontSize: '10px' }}>CPU CPs</span>
              <div style={{ marginTop: '4px' }}>
                {Array.from(cpuLapTimerState.checkpointsPassed.entries()).map(([id, passed]) => (
                  <span key={id} style={{ color: passed ? '#00ff00' : '#666', marginRight: '8px' }}>
                    {id}:{passed ? 'Y' : 'N'}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="controls-hint" style={{
        marginTop: '10px',
        padding: '10px',
        backgroundColor: '#222',
        color: '#aaa',
        fontFamily: 'sans-serif',
        fontSize: '14px',
        borderRadius: '4px',
        textAlign: 'center',
      }}>
        <strong>Controls:</strong> Arrow Keys or WASD to drive |
        <span style={{ color: '#e74c3c' }}> Red = You</span> |
        <span style={{ color: mlDriverRef.current ? '#9b59b6' : '#3498db' }}>
          {mlDriverRef.current ? ' Purple = ML CPU' : ' Blue = CPU'}
        </span>
      </div>
      {showDebug && (
        <div className="debug-info" style={{
          marginTop: '10px',
          padding: '10px',
          backgroundColor: '#333',
          color: '#fff',
          fontFamily: 'monospace',
          fontSize: '12px',
          borderRadius: '4px',
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '10px',
        }}>
          <div>
            <div style={{ color: '#e74c3c', fontWeight: 'bold' }}>PLAYER DEBUG</div>
            <div>Position: ({playerCar.position.x.toFixed(1)}, {playerCar.position.y.toFixed(1)})</div>
            <div>Angle: {(playerCar.angle * 180 / Math.PI).toFixed(1)}deg</div>
            <div>Speed: {playerCar.speed.toFixed(1)} px/s</div>
            <div>Wall Collision: {playerCar.isColliding ? 'Yes' : 'No'}</div>
          </div>
          <div>
            <div style={{ color: mlDriverRef.current ? '#9b59b6' : '#3498db', fontWeight: 'bold' }}>
              CPU DEBUG {mlDriverRef.current ? '(ML)' : '(Waypoint)'}
            </div>
            <div>Position: ({cpuCar.position.x.toFixed(1)}, {cpuCar.position.y.toFixed(1)})</div>
            <div>Angle: {(cpuCar.angle * 180 / Math.PI).toFixed(1)}deg</div>
            <div>Speed: {cpuCar.speed.toFixed(1)} px/s</div>
            <div>Wall Collision: {cpuCar.isColliding ? 'Yes' : 'No'}</div>
            {mlDriverRef.current && mlDebugInfo ? (
              <>
                <div>Action: {mlDebugInfo.actionName} ({mlDebugInfo.lastAction})</div>
                <div>Model: {mlDebugInfo.usingFallback ? 'Fallback' : 'Trained'}</div>
                <div>Checkpoints: {mlDebugInfo.checkpointsPassed}</div>
              </>
            ) : (
              <div>Target WP: {cpuDriverRef.current.targetWaypoint}/{cpuDriverRef.current.waypoints.length}</div>
            )}
          </div>
          <div style={{ gridColumn: '1 / -1' }}>
            <div>Car Collision: {carsColliding ? 'COLLIDING!' : 'No'}</div>
          </div>
        </div>
      )}
    </div>
  );
}

export default GameCanvas;
