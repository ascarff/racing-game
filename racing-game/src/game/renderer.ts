/**
 * Canvas rendering utilities for the 2D racing game
 */

import type { Point, TrackData, TrackRenderConfig } from './types';
import { defaultTrackRenderConfig } from './track';

/**
 * Clear the canvas with the background color
 */
export function clearCanvas(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  config: TrackRenderConfig = defaultTrackRenderConfig
): void {
  ctx.fillStyle = config.backgroundColor;
  ctx.fillRect(0, 0, width, height);
}

/**
 * Draw a filled polygon from an array of points
 */
export function drawFilledPolygon(
  ctx: CanvasRenderingContext2D,
  points: Point[],
  fillColor: string
): void {
  if (points.length < 3) return;

  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.closePath();
  ctx.fillStyle = fillColor;
  ctx.fill();
}

/**
 * Draw a polygon outline from an array of points
 */
export function drawPolygonOutline(
  ctx: CanvasRenderingContext2D,
  points: Point[],
  strokeColor: string,
  lineWidth: number
): void {
  if (points.length < 2) return;

  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.closePath();
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
}

/**
 * Draw a line between two points
 */
export function drawLine(
  ctx: CanvasRenderingContext2D,
  start: Point,
  end: Point,
  color: string,
  lineWidth: number
): void {
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
}

/**
 * Draw the track surface (the drivable area between inner and outer boundaries)
 */
export function drawTrackSurface(
  ctx: CanvasRenderingContext2D,
  trackData: TrackData,
  config: TrackRenderConfig = defaultTrackRenderConfig
): void {
  const { boundaries } = trackData;

  // Use compositing to create the track surface as a ring
  ctx.save();

  // Draw outer ellipse filled
  drawFilledPolygon(ctx, boundaries.outer, config.trackColor);

  // Cut out inner ellipse using destination-out compositing
  ctx.globalCompositeOperation = 'destination-out';
  drawFilledPolygon(ctx, boundaries.inner, 'black');

  ctx.restore();

  // Now draw the grass in the center (inner area)
  drawFilledPolygon(ctx, boundaries.inner, config.grassColor);
}

/**
 * Draw track boundaries (white lines on edges)
 */
export function drawTrackBoundaries(
  ctx: CanvasRenderingContext2D,
  trackData: TrackData,
  config: TrackRenderConfig = defaultTrackRenderConfig
): void {
  const { boundaries } = trackData;

  // Draw outer boundary
  drawPolygonOutline(ctx, boundaries.outer, config.boundaryColor, config.boundaryWidth);

  // Draw inner boundary
  drawPolygonOutline(ctx, boundaries.inner, config.boundaryColor, config.boundaryWidth);
}

/**
 * Draw the start/finish line with a checkered pattern
 */
export function drawStartFinishLine(
  ctx: CanvasRenderingContext2D,
  trackData: TrackData,
  config: TrackRenderConfig = defaultTrackRenderConfig
): void {
  const { startFinishLine } = trackData;
  const { start, end } = startFinishLine;

  // Calculate line properties
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.sqrt(dx * dx + dy * dy);
  const angle = Math.atan2(dy, dx);

  // Draw checkered pattern
  const numSquares = 8;
  const squareSize = length / numSquares;

  ctx.save();
  ctx.translate(start.x, start.y);
  ctx.rotate(angle);

  for (let i = 0; i < numSquares; i++) {
    ctx.fillStyle = i % 2 === 0 ? '#ffffff' : '#000000';
    ctx.fillRect(
      i * squareSize,
      -config.startFinishWidth / 2,
      squareSize,
      config.startFinishWidth
    );
  }

  ctx.restore();

  // Draw red lines on the edges of the start/finish area
  const perpX = -dy / length * (config.startFinishWidth / 2 + 2);
  const perpY = dx / length * (config.startFinishWidth / 2 + 2);

  drawLine(
    ctx,
    { x: start.x + perpX, y: start.y + perpY },
    { x: end.x + perpX, y: end.y + perpY },
    config.startFinishColor,
    2
  );
  drawLine(
    ctx,
    { x: start.x - perpX, y: start.y - perpY },
    { x: end.x - perpX, y: end.y - perpY },
    config.startFinishColor,
    2
  );
}

/**
 * Draw checkpoints (optional, for debugging)
 */
export function drawCheckpoints(
  ctx: CanvasRenderingContext2D,
  trackData: TrackData,
  visible: boolean = false
): void {
  if (!visible) return;

  const { checkpoints } = trackData;

  ctx.save();
  ctx.setLineDash([5, 5]);

  for (const checkpoint of checkpoints) {
    drawLine(
      ctx,
      checkpoint.line.start,
      checkpoint.line.end,
      '#ffff00',
      2
    );

    // Draw checkpoint number
    const midX = (checkpoint.line.start.x + checkpoint.line.end.x) / 2;
    const midY = (checkpoint.line.start.y + checkpoint.line.end.y) / 2;
    ctx.fillStyle = '#ffff00';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`CP${checkpoint.id}`, midX, midY);
  }

  ctx.restore();
}

/**
 * Draw the complete track
 */
export function drawTrack(
  ctx: CanvasRenderingContext2D,
  trackData: TrackData,
  config: TrackRenderConfig = defaultTrackRenderConfig,
  showCheckpoints: boolean = false
): void {
  // Clear canvas
  clearCanvas(ctx, trackData.dimensions.width, trackData.dimensions.height, config);

  // Draw track surface
  drawTrackSurface(ctx, trackData, config);

  // Draw boundaries
  drawTrackBoundaries(ctx, trackData, config);

  // Draw start/finish line
  drawStartFinishLine(ctx, trackData, config);

  // Draw checkpoints (if enabled)
  drawCheckpoints(ctx, trackData, showCheckpoints);
}

/**
 * Draw a vehicle (placeholder for Story 2)
 * @param ctx - Canvas rendering context
 * @param position - Vehicle position
 * @param angle - Vehicle rotation angle in radians
 * @param color - Vehicle color
 */
export function drawVehicle(
  ctx: CanvasRenderingContext2D,
  position: Point,
  angle: number,
  color: string = '#ff0000'
): void {
  const width = 30;
  const height = 50;

  ctx.save();
  ctx.translate(position.x, position.y);
  ctx.rotate(angle + Math.PI / 2); // Adjust so 0 angle faces up

  // Draw car body
  ctx.fillStyle = color;
  ctx.fillRect(-width / 2, -height / 2, width, height);

  // Draw front indicator
  ctx.fillStyle = '#ffff00';
  ctx.fillRect(-width / 4, -height / 2, width / 2, 8);

  ctx.restore();
}
