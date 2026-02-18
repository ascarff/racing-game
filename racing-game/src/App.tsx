import { useReducer, useCallback, useState, useEffect } from 'react';
import { GameCanvas, type RaceFinishData } from './components/GameCanvas';
import { MainMenu } from './components/MainMenu';
import { RaceResults } from './components/RaceResults';
import { gameStateReducer, initialAppState } from './game/gameState';
import {
  loadBestTimes,
  saveBestTimes,
  clearBestTimes,
  updateBestTimes,
  type BestTimes,
} from './game/storage';
import './App.css';

function App() {
  const [appState, dispatch] = useReducer(gameStateReducer, initialAppState);
  const [showDebug, setShowDebug] = useState(false);
  const [bestTimes, setBestTimes] = useState<BestTimes>(() => loadBestTimes());
  const [newRecord, setNewRecord] = useState<{ lap: boolean; race: boolean }>({
    lap: false,
    race: false,
  });
  // Store race results for the results screen
  const [raceResults, setRaceResults] = useState<RaceFinishData | null>(null);

  // Load best times on mount
  useEffect(() => {
    const loaded = loadBestTimes();
    setBestTimes(loaded);
    // Sync with app state
    if (loaded.bestLapTime !== null) {
      dispatch({ type: 'UPDATE_BEST_TIME', time: loaded.bestLapTime });
    }
  }, []);

  const handleStartRace = useCallback(() => {
    // Reset new record flags and race results when starting a new race
    setNewRecord({ lap: false, race: false });
    setRaceResults(null);
    dispatch({ type: 'START_RACE' });
  }, []);

  const handleReturnToMenu = useCallback(() => {
    // Reset new record flags and race results when returning to menu
    setNewRecord({ lap: false, race: false });
    setRaceResults(null);
    dispatch({ type: 'RETURN_TO_MENU' });
  }, []);

  const handleRaceFinished = useCallback((data: RaceFinishData) => {
    if (data.playerLapTimes.length === 0) return;

    // Store race results for the results screen
    setRaceResults(data);

    // Find best lap time from this race
    const bestLapThisRace = Math.min(...data.playerLapTimes);
    // Total race time is already calculated
    const totalRaceTime = data.playerTotalTime;

    // Update best times
    const { times: updatedTimes, newLapRecord, newRaceRecord } = updateBestTimes(
      bestTimes,
      bestLapThisRace,
      totalRaceTime
    );

    // Save if any records were broken
    if (newLapRecord || newRaceRecord) {
      setBestTimes(updatedTimes);
      saveBestTimes(updatedTimes);
      setNewRecord({ lap: newLapRecord, race: newRaceRecord });

      // Update app state with new best lap time
      if (newLapRecord) {
        dispatch({ type: 'UPDATE_BEST_TIME', time: bestLapThisRace });
      }
    }

    // Transition to results screen
    dispatch({ type: 'FINISH_RACE', time: totalRaceTime });
  }, [bestTimes]);

  const handleClearBestTimes = useCallback(() => {
    clearBestTimes();
    setBestTimes({ bestLapTime: null, bestRaceTime: null });
    dispatch({ type: 'UPDATE_BEST_TIME', time: Infinity }); // Reset to no best time
  }, []);

  // Render based on current screen
  if (appState.currentScreen === 'menu') {
    return (
      <MainMenu
        bestTime={bestTimes.bestLapTime}
        onStartRace={handleStartRace}
        onClearBestTimes={handleClearBestTimes}
      />
    );
  }

  // Results screen - show after race is finished
  if (appState.currentScreen === 'results' && raceResults) {
    return (
      <RaceResults
        playerPosition={raceResults.playerPosition}
        totalTime={raceResults.playerTotalTime}
        lapTimes={raceResults.playerLapTimes}
        cpuLapTimes={raceResults.cpuLapTimes}
        cpuTotalTime={raceResults.cpuTotalTime}
        onRaceAgain={handleStartRace}
        onMainMenu={handleReturnToMenu}
      />
    );
  }

  // Racing screen
  return (
    <div className="app">
      <header className="app-header">
        <h1>2D Racing Game</h1>
        <div className="header-controls">
          <button className="menu-button" onClick={handleReturnToMenu}>
            Back to Menu
          </button>
          <label className="debug-toggle">
            <input
              type="checkbox"
              checked={showDebug}
              onChange={(e) => setShowDebug(e.target.checked)}
            />
            Show Debug Info
          </label>
        </div>
      </header>
      <main className="app-main">
        <GameCanvas
          showDebug={showDebug}
          bestLapTime={bestTimes.bestLapTime}
          onRaceFinished={handleRaceFinished}
          newLapRecord={newRecord.lap}
        />
      </main>
      <footer className="app-footer">
        <p>Use arrow keys or WASD to control the car</p>
      </footer>
    </div>
  );
}

export default App;
