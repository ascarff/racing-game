/**
 * Keyboard input handling for the 2D racing game
 * Tracks which keys are currently pressed and provides input state for the game loop
 */

/**
 * Input state representing current control inputs
 */
export interface InputState {
  /** Accelerate (up arrow or W) */
  accelerate: boolean;
  /** Brake/reverse (down arrow or S) */
  brake: boolean;
  /** Steer left (left arrow or A) */
  steerLeft: boolean;
  /** Steer right (right arrow or D) */
  steerRight: boolean;
}

/**
 * Keys that map to game controls
 */
const KEY_MAPPINGS = {
  accelerate: ['ArrowUp', 'KeyW'],
  brake: ['ArrowDown', 'KeyS'],
  steerLeft: ['ArrowLeft', 'KeyA'],
  steerRight: ['ArrowRight', 'KeyD'],
} as const;

/**
 * Set of currently pressed keys
 */
const pressedKeys = new Set<string>();

/**
 * Whether the input system has been initialized
 */
let isInitialized = false;

/**
 * Handle keydown events
 */
function handleKeyDown(event: KeyboardEvent): void {
  // Prevent default for game control keys (stops page scrolling)
  const isGameKey = Object.values(KEY_MAPPINGS)
    .flat()
    .includes(event.code as any);

  if (isGameKey) {
    event.preventDefault();
    pressedKeys.add(event.code);
  }
}

/**
 * Handle keyup events
 */
function handleKeyUp(event: KeyboardEvent): void {
  pressedKeys.delete(event.code);
}

/**
 * Handle window blur (release all keys when window loses focus)
 */
function handleBlur(): void {
  pressedKeys.clear();
}

/**
 * Initialize the input system
 * Adds event listeners for keyboard input
 * Call this once when the game starts
 */
export function initInput(): void {
  if (isInitialized) return;

  window.addEventListener('keydown', handleKeyDown);
  window.addEventListener('keyup', handleKeyUp);
  window.addEventListener('blur', handleBlur);

  isInitialized = true;
}

/**
 * Cleanup the input system
 * Removes event listeners
 * Call this when the game is unmounted
 */
export function cleanupInput(): void {
  if (!isInitialized) return;

  window.removeEventListener('keydown', handleKeyDown);
  window.removeEventListener('keyup', handleKeyUp);
  window.removeEventListener('blur', handleBlur);

  pressedKeys.clear();
  isInitialized = false;
}

/**
 * Check if any of the specified keys are pressed
 */
function isAnyKeyPressed(keys: readonly string[]): boolean {
  return keys.some(key => pressedKeys.has(key));
}

/**
 * Get the current input state
 * Call this each frame to get player input
 */
export function getInputState(): InputState {
  return {
    accelerate: isAnyKeyPressed(KEY_MAPPINGS.accelerate),
    brake: isAnyKeyPressed(KEY_MAPPINGS.brake),
    steerLeft: isAnyKeyPressed(KEY_MAPPINGS.steerLeft),
    steerRight: isAnyKeyPressed(KEY_MAPPINGS.steerRight),
  };
}

/**
 * Check if a specific control is active
 * Useful for checking individual controls without getting full state
 */
export function isControlActive(control: keyof InputState): boolean {
  return isAnyKeyPressed(KEY_MAPPINGS[control]);
}

/**
 * Get raw pressed keys (for debugging)
 */
export function getPressedKeys(): string[] {
  return Array.from(pressedKeys);
}
