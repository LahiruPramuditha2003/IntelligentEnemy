/**
 * brain.js — AI Brain base class and manager.
 * 
 * DESIGN PATTERN: Strategy Pattern
 * 
 * The Strategy Pattern defines a family of algorithms (AI brains), encapsulates
 * each one, and makes them interchangeable. The game code doesn't need to know
 * which specific brain is running — it just calls decide() and gets an action.
 * 
 * This is perfect for our use case because:
 * 1. We can hot-swap between Random, Heuristic, Q-Learning, DQN brains
 * 2. We can compare their performance under identical conditions
 * 3. Adding a new brain type requires zero changes to existing game code
 * 
 * Interface contract:
 *   decide(gameState) → action     // Choose what to do
 *   onDeath(gameState)             // Learn from failure
 *   onSurvive(gameState)           // Learn from success
 *   reset()                        // Prepare for new round
 */

export class AIBrain {
    /**
     * @param {string} name - Display name for this brain (shown in UI)
     */
    constructor(name) {
        this.name = name;
    }

    /**
     * Given the current game state, choose an action.
     * This is the core method every brain must implement.
     * 
     * @param {Object} gameState - Structured snapshot of the game:
     *   {
     *     enemyX, enemyY, enemyVX, enemyVY,    // Enemy position & velocity
     *     enemySpeed, enemyIsJumping,            // Enemy state
     *     stones: [{x, y, vy}, ...],             // All active stones
     *     nearestStoneX, nearestStoneY,          // Nearest stone position
     *     nearestStoneVY, nearestStoneDist,      // Nearest stone velocity & distance
     *     canvasWidth, groundY                   // Environment bounds
     *   }
     * @returns {string} One of the ACTION_LIST values
     */
    decide(gameState) {
        throw new Error('decide() must be implemented by subclass');
    }

    /**
     * Called when the enemy is hit and dies.
     * This is the primary learning signal — the brain receives the final
     * game state and can update its strategy accordingly.
     * 
     * @param {Object} gameState - Game state at the moment of death
     */
    onDeath(gameState) {
        // Override in subclass for learning
    }

    /**
     * Called when the enemy survives (e.g., end of a timed round).
     * Positive reinforcement signal.
     * 
     * @param {Object} gameState - Current game state
     */
    onSurvive(gameState) {
        // Override in subclass for learning
    }

    /**
     * Reset internal state for a new round.
     * Called at the start of each round — not between decisions.
     */
    reset() {
        // Override in subclass if needed
    }
}
