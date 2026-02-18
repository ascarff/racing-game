/**
 * Game module exports
 * Re-exports all game-related types, data, and utilities
 */

// Types
export type {
  Point,
  LineSegment,
  TrackBoundary,
  StartFinishLine,
  Checkpoint,
  TrackData,
  TrackRenderConfig,
  GameState,
  VehicleState,
} from './types';

// Track data and utilities
export {
  trackData,
  defaultTrackRenderConfig,
  getTrackCenterPoint,
  isPointOnTrack,
  getTrackNormal,
} from './track';

// Renderer utilities
export {
  clearCanvas,
  drawFilledPolygon,
  drawPolygonOutline,
  drawLine,
  drawTrackSurface,
  drawTrackBoundaries,
  drawStartFinishLine,
  drawCheckpoints,
  drawTrack,
  drawVehicle,
} from './renderer';

// Game state management
export type {
  GameScreen,
  AppState,
  GameAction,
} from './gameState';

export {
  initialAppState,
  formatTime,
  gameStateReducer,
} from './gameState';

// CPU Driver (AI opponent)
export type {
  CPUDriver,
  CPUDriverConfig,
} from './cpuDriver';

export {
  createCPUDriver,
  updateCPUDriver,
  resetCPUDriver,
  getCPUCarState,
  getTargetWaypoint,
  generateWaypoints,
  drawWaypoints,
  defaultCPUDriverConfig,
} from './cpuDriver';

// ML Driver (ML-based AI opponent)
export type {
  MLDriver,
} from './mlDriver';

export {
  createMLDriver,
  updateMLDriver,
  resetMLDriver,
  getMLCarState,
  loadMLModel,
  isUsingFallback,
  getMLDriverDebugInfo,
  drawMLDriverDebug,
  actionToInput,
} from './mlDriver';
