# Simple 2D Racing Game - User Stories

## Assumptions

Before breaking down this deliverable, I'm making the following assumptions:

1. **Platform:** Browser-based game (HTML5 Canvas or similar) - no native app required
2. **Simplicity:** Top-down or side-view 2D, single track to start, basic car physics
3. **Single player focus:** Player races against CPU opponent(s), no multiplayer
4. **ML approach:** Reinforcement learning (e.g., Q-learning or simple neural network) for CPU training
5. **Persistence:** Best times stored locally (localStorage) or simple backend - not a full user account system
6. **Track:** Pre-defined track(s), not user-generated
7. **"Simple" means:** Minimal UI, basic graphics, core racing mechanics only

---

## Must-Have Stories

### Story 1: Basic Game Canvas & Rendering ✅

**User Story:** As a player, I want to see a 2D race track rendered on screen so that I have a visual environment to race in.

**Complexity:** Cat (3)

**Role:** frontend

**Dependencies:** None

**Acceptance Criteria:**
- [x] Game canvas renders at appropriate resolution (responsive or fixed)
- [x] Track boundaries are clearly visible
- [x] Start/finish line is marked
- [x] Track has at least one turn (not just a straight line)
- [x] Background and track are visually distinct
- [x] Canvas renders without errors on modern browsers (Chrome, Firefox, Safari)

---

### Story 2: Player Car Controls ✅

**User Story:** As a player, I want to control my car using keyboard inputs so that I can navigate the track.

**Complexity:** Big Dog (5)

**Role:** frontend

**Dependencies:** Story 1

**Acceptance Criteria:**
- [x] Car accelerates when up arrow/W is pressed
- [x] Car brakes/reverses when down arrow/S is pressed
- [x] Car steers left/right with arrow keys or A/D
- [x] Car has momentum (doesn't stop instantly)
- [x] Car cannot drive through track boundaries (collision detection)
- [x] Car speed is capped at a maximum value
- [x] Controls are responsive with no perceptible input lag

---

### Story 3: Lap Timing System ✅

**User Story:** As a player, I want my lap times tracked so that I know how fast I completed each lap.

**Complexity:** Cat (3)

**Role:** frontend

**Dependencies:** Story 1, Story 2

**Acceptance Criteria:**
- [x] Timer starts when race begins
- [x] Lap time recorded when car crosses finish line
- [x] Current lap time displayed during race
- [x] Lap only counts if player completes the full track (checkpoints or direction detection)
- [x] Timer displays in MM:SS.ms format
- [x] Timer pauses if game is paused

---

### Story 4: Race Start & Finish Flow ✅

**User Story:** As a player, I want a clear start and finish to each race so that I know when to begin and when I've completed the race.

**Complexity:** Cat (3)

**Role:** frontend

**Dependencies:** Story 2, Story 3

**Acceptance Criteria:**
- [x] Countdown (3-2-1-GO) before race starts
- [x] Player cannot move during countdown
- [x] Race ends after completing set number of laps (default: 3)
- [x] End screen shows final time and lap breakdown
- [x] Option to restart race from end screen
- [x] Option to return to main menu from end screen

---

### Story 5: Basic CPU Car (Non-ML) ✅

**User Story:** As a player, I want a CPU opponent on the track so that I have competition while the ML model is being developed.

**Complexity:** Big Dog (5)

**Role:** frontend

**Dependencies:** Story 1, Story 2

**Acceptance Criteria:**
- [x] CPU car renders on track with distinct appearance from player
- [x] CPU follows the track using waypoints or path-following
- [x] CPU car respects same physics as player (speed limits, collision)
- [x] CPU car and player car can collide (basic collision response)
- [x] CPU completes laps and has recorded lap times
- [x] CPU provides reasonable competition (not too easy, not impossible)

---

### Story 6: ML Training Environment ✅

**User Story:** As a developer, I want a training environment for the CPU so that the ML model can learn to race through repeated simulations.

**Complexity:** Deer (8)

**Role:** data

**Dependencies:** Story 1, Story 2

**Acceptance Criteria:**
- [x] Game can run in headless/accelerated mode (faster than real-time)
- [x] State can be extracted (car position, velocity, track boundaries, checkpoints)
- [x] Actions can be programmatically input (accelerate, brake, steer)
- [x] Reward function defined (progress on track, penalize wall hits, reward speed)
- [x] Episode resets when car crashes or completes lap
- [x] Training data/metrics can be logged
- [x] Environment supports running multiple parallel instances

---

### Story 7: ML Model Training Pipeline ✅

**User Story:** As a developer, I want to train an ML model to control the CPU car so that it learns optimal racing behavior.

**Complexity:** Cow (13)

**Role:** data

**Dependencies:** Story 6

**Acceptance Criteria:**
- [x] Model architecture defined (e.g., simple neural network, Q-table)
- [x] Training loop implemented with configurable episodes
- [x] Model improves over training (measurable lap time reduction)
- [x] Trained model can be saved/exported
- [x] Training progress can be monitored (loss, reward, lap times)
- [x] Hyperparameters are configurable
- [x] Model achieves reasonable performance (completes laps without excessive crashes)

---

### Story 8: ML-Controlled CPU Integration ✅

**User Story:** As a player, I want to race against an ML-trained CPU opponent so that I experience intelligent competition.

**Complexity:** Big Dog (5)

**Role:** frontend, data

**Dependencies:** Story 5, Story 7

**Acceptance Criteria:**
- [x] Trained model loads at game start
- [x] CPU car controlled by model predictions in real-time
- [x] Model runs efficiently (no frame drops during inference)
- [x] CPU behavior is visibly "intelligent" (follows racing line, avoids walls)
- [x] Fallback to basic CPU if model fails to load
- [x] CPU difficulty feels balanced for casual player

---

### Story 9: Best Times Persistence ✅

**User Story:** As a player, I want my best lap times saved so that I can track my improvement over time.

**Complexity:** Cat (3)

**Role:** frontend

**Dependencies:** Story 3

**Acceptance Criteria:**
- [x] Best lap time saved after each race
- [x] Best time persists between browser sessions (localStorage)
- [x] Best time displayed on main menu
- [x] Best time displayed during race (as target to beat)
- [x] New record notification when best time is beaten
- [x] Clear/reset best times option available

---

### Story 10: Main Menu ✅

**User Story:** As a player, I want a main menu so that I can start races and view my best times.

**Complexity:** Mouse (1)

**Role:** frontend

**Dependencies:** Story 1

**Acceptance Criteria:**
- [x] Menu displays on game load
- [x] "Start Race" button begins the game
- [x] Best time displayed on menu
- [x] Menu is navigable with keyboard or mouse
- [x] Clean, simple visual design

---

## Nice-to-Have Stories

### Story 11: Multiple Difficulty Levels

**User Story:** As a player, I want to choose CPU difficulty so that I can have appropriate challenge for my skill level.

**Complexity:** Cat (3)

**Role:** frontend, data

**Dependencies:** Story 8

**Acceptance Criteria:**
- [ ] Easy, Medium, Hard options available
- [ ] Difficulty affects CPU speed/performance
- [ ] Difficulty selection on main menu
- [ ] Best times tracked separately per difficulty
- [ ] Default difficulty is Medium

---

### Story 12: Leaderboard (Backend)

**User Story:** As a player, I want to see a global leaderboard so that I can compare my times with other players.

**Complexity:** Deer (8)

**Role:** backend, frontend

**Dependencies:** Story 9

**Acceptance Criteria:**
- [ ] Backend API to submit and retrieve times
- [ ] Player can enter name/initials with their time
- [ ] Top 10 times displayed on leaderboard screen
- [ ] Leaderboard accessible from main menu
- [ ] Handles network errors gracefully
- [ ] Basic anti-cheat (server validates reasonable times)

---

### Story 13: Sound Effects & Music

**User Story:** As a player, I want sound effects and background music so that the game feels more engaging.

**Complexity:** Cat (3)

**Role:** frontend

**Dependencies:** Story 2, Story 4

**Acceptance Criteria:**
- [ ] Engine sound that changes with speed
- [ ] Collision/crash sound effect
- [ ] Countdown beeps
- [ ] Background music during race
- [ ] Mute/volume controls available
- [ ] Sounds don't overlap or cause audio glitches

---

### Story 14: Multiple Tracks

**User Story:** As a player, I want multiple tracks to choose from so that I have variety in gameplay.

**Complexity:** Big Dog (5)

**Role:** frontend, data

**Dependencies:** Story 1, Story 8

**Acceptance Criteria:**
- [ ] At least 2-3 different track layouts
- [ ] Track selection on main menu
- [ ] Best times saved per track
- [ ] ML model works on all tracks (or separate models per track)
- [ ] Visual distinction between tracks

---

### Story 15: Ghost Replay

**User Story:** As a player, I want to race against a ghost of my best lap so that I can see exactly where to improve.

**Complexity:** Big Dog (5)

**Role:** frontend

**Dependencies:** Story 9

**Acceptance Criteria:**
- [ ] Best lap inputs/positions recorded
- [ ] Ghost car renders semi-transparently during race
- [ ] Ghost replays best lap in real-time alongside player
- [ ] Option to toggle ghost on/off
- [ ] Ghost updates when new best time is set

---

## Summary

| Category | Count | Total Complexity | Status |
|----------|-------|------------------|--------|
| Must-Have | 10 stories | 49 points | ✅ Complete |
| Nice-to-Have | 5 stories | 24 points | Not started |
| **Total** | **15 stories** | **73 points** | |

### Role Breakdown

| Role | Stories |
|------|---------|
| Frontend | 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15 |
| Data | 6, 7, 8, 11, 14 |
| Backend | 12 |

### Suggested Implementation Order

**Phase 1 - Core Game (Stories 1, 2, 3, 4, 10):**
Build the playable racing game foundation. Player can drive, complete laps, and see times.

**Phase 2 - Basic Competition (Stories 5, 9):**
Add simple CPU opponent and time persistence. Game is now complete as a simple racing experience.

**Phase 3 - ML Development (Stories 6, 7):**
Build training environment and train the ML model. Can be done in parallel with Phase 2.

**Phase 4 - ML Integration (Story 8):**
Replace basic CPU with ML-trained opponent. Core deliverable complete.

**Phase 5 - Polish (Stories 11-15):**
Add difficulty levels, leaderboard, sound, tracks, and ghost replay based on priority.

### Open Questions for Stakeholder

1. **Track design:** Should tracks be hand-drawn assets or procedurally generated shapes?
2. **ML algorithm preference:** Any preference on RL approach (Q-learning, PPO, genetic algorithm)?
3. **Leaderboard priority:** Is global leaderboard a must-have or nice-to-have? (Requires backend infrastructure)
4. **Target platforms:** Browser only, or also desktop (Electron) / mobile?
5. **Art style:** Minimalist shapes, pixel art, or vector graphics?
6. **Number of laps:** Configurable or fixed at 3?
7. **Multiple CPU opponents:** Race against 1 CPU or multiple?
