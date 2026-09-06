/**
 * game.js — Game State Manager.
 * 
 * This is the central coordinator for the game. It:
 * 1. Owns all game entities (enemy, stones)
 * 2. Manages round transitions (playing → defeated → new round)
 * 3. Drives the AI decision-making cycle
 * 4. Extracts game state for the AI brain (state representation)
 * 5. Tracks statistics for measuring AI performance
 * 
 * STATE REPRESENTATION (Critical AI Concept):
 * The getGameState() method constructs the "observation" that the AI
 * sees. What information we include here directly affects what the AI
 * can learn. This is one of the most important design decisions in RL:
 * - Too little info → agent can't make good decisions (partial observability)
 * - Too much info → agent is slow to learn (curse of dimensionality)
 * - Right features → agent learns efficiently
 */

import { CONFIG } from './config.js';
import { Enemy } from './enemy.js';
import { StoneManager } from './stone.js';
import { checkStoneEnemyCollision } from './physics.js';

export class Game {
    constructor() {
        this.enemy = new Enemy();
        this.stoneManager = new StoneManager();

        // Round management
        this.round = 1;
        this.state = 'PLAYING';     // 'PLAYING' | 'ROUND_OVER'
        this.roundOverTime = 0;

        // Performance statistics — essential for measuring AI improvement
        this.stats = {
            totalRounds: 0,
            playerWins: 0,
            currentSurvivalTime: 0,     // seconds survived this round
            longestSurvival: 0,         // best survival time across all rounds
            recentSurvivals: [],        // last N survival times (for rolling average)
        };

        // AI control
        this.aiBrain = null;
        this.lastDecisionTime = 0;

        // Visual / Audio FX Callbacks
        this.onHitCallback = null;
        this.onGroundHitCallback = null;
    }

    /**
     * Set the active AI brain (hot-swappable via Strategy Pattern).
     * @param {AIBrain} brain
     */
    setAIBrain(brain) {
        this.aiBrain = brain;
    }

    /**
     * Main update — called once per frame.
     * 
     * Game loop order matters:
     * 1. AI decides an action (based on current state)
     * 2. Physics update (apply action + gravity + movement)
     * 3. Collision check (did a stone hit the enemy?)
     * 4. State transition (alive → dead → new round)
     * 
     * @param {number} dt - Delta time in seconds
     * @param {number} currentTime - Current timestamp in ms
     */
    update(dt, currentTime) {
        // Handle round-over pause
        if (this.state === 'ROUND_OVER') {
            if (currentTime - this.roundOverTime >= CONFIG.ROUND_RESET_DELAY) {
                this.startNewRound();
            }
            return;
        }

        if (this.state !== 'PLAYING') return;

        // Track how long the enemy has survived
        this.stats.currentSurvivalTime += dt;

        // --- AI Decision ---
        // The AI makes a decision every AI_DECISION_INTERVAL milliseconds.
        // WHY NOT EVERY FRAME?
        // 1. Decisions every frame (16ms) would make the enemy jitter wildly
        // 2. Real-world agents don't act at infinite frequency
        // 3. For Q-Learning, fewer decisions = smaller state-action history = faster learning
        if (this.aiBrain && currentTime - this.lastDecisionTime >= CONFIG.AI_DECISION_INTERVAL) {
            const gameState = this.getGameState();
            const action = this.aiBrain.decide(gameState);
            this.enemy.applyAction(action);
            this.lastDecisionTime = currentTime;
        }

        // --- Physics Update ---
        this.enemy.update(dt);
        this.stoneManager.update(dt, this.onGroundHitCallback);

        // --- Collision Detection ---
        for (const stone of this.stoneManager.getActiveStones()) {
            if (checkStoneEnemyCollision(stone, this.enemy)) {
                this.onEnemyHit(currentTime);
                stone.active = false;
                break;  // One hit is enough
            }
        }
    }

    /**
     * Handle the enemy being hit by a stone.
     * @param {number} currentTime
     */
    onEnemyHit(currentTime) {
        this.enemy.die();
        this.state = 'ROUND_OVER';
        this.roundOverTime = currentTime;

        // Trigger defeat explosion callback
        if (this.onHitCallback) {
            this.onHitCallback(this.enemy.x, this.enemy.y);
        }

        // Update statistics
        this.stats.playerWins++;
        this.stats.totalRounds++;

        if (this.stats.currentSurvivalTime > this.stats.longestSurvival) {
            this.stats.longestSurvival = this.stats.currentSurvivalTime;
        }

        // Track recent survivals (keep last 20 for rolling average)
        this.stats.recentSurvivals.push(this.stats.currentSurvivalTime);
        if (this.stats.recentSurvivals.length > 20) {
            this.stats.recentSurvivals.shift();
        }

        // Notify AI brain — THIS IS WHERE LEARNING HAPPENS
        if (this.aiBrain) {
            this.aiBrain.onDeath(this.getGameState());
        }
    }

    /**
     * Start a new round after the pause.
     */
    startNewRound() {
        this.round++;
        this.state = 'PLAYING';
        this.enemy.reset();
        this.stoneManager.reset();
        this.stats.currentSurvivalTime = 0;
        this.lastDecisionTime = 0;

        // Let the brain prepare for the new round
        if (this.aiBrain) {
            this.aiBrain.reset();
        }
    }

    /**
     * Spawn a stone at the given X position (player action).
     * @param {number} x - Horizontal position
     * @param {number} currentTime - Current timestamp in ms
     */
    spawnStone(x, currentTime) {
        if (this.state !== 'PLAYING') return;
        this.stoneManager.spawn(x, currentTime);
    }

    /**
     * Extract the current game state for the AI brain.
     * 
     * This is the AI's "observation" — everything the agent knows about
     * the world. We include both raw values and derived features:
     * 
     * Raw values:    enemyX, enemyVX, stone positions/velocities
     * Derived:       nearestStoneX, nearestStoneDist (hand-crafted features)
     * 
     * LATER (Phase 3-4): We'll experiment with different state representations:
     * - Normalized values (0-1 range) — helps neural networks learn faster
     * - Grid-based discretization — required for tabular Q-Learning
     * - Raw pixels — for CNN-based approaches (very advanced)
     * 
     * @returns {Object} Structured game state snapshot
     */
    getGameState() {
        const activeStones = this.stoneManager.getActiveStones();

        // Find the nearest stone to the enemy (most immediate threat)
        let nearestStone = null;
        let nearestDist = Infinity;

        const enemyCenterX = this.enemy.x + this.enemy.width / 2;
        const enemyCenterY = this.enemy.y + this.enemy.height / 2;

        for (const stone of activeStones) {
            const dx = stone.x - enemyCenterX;
            const dy = stone.y - enemyCenterY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < nearestDist) {
                nearestDist = dist;
                nearestStone = stone;
            }
        }

        return {
            // Enemy state
            enemyX: this.enemy.x,
            enemyY: this.enemy.y,
            enemyVX: this.enemy.vx,
            enemyVY: this.enemy.vy,
            enemySpeed: this.enemy.speed,
            enemyIsJumping: this.enemy.isJumping,

            // All active stones (for brains that want full information)
            stones: activeStones.map(s => ({
                x: s.x,
                y: s.y,
                vy: s.vy,
            })),

            // Nearest stone (convenience features for simpler brains)
            nearestStoneX: nearestStone ? nearestStone.x : -1,
            nearestStoneY: nearestStone ? nearestStone.y : -1,
            nearestStoneVY: nearestStone ? nearestStone.vy : 0,
            nearestStoneDist: nearestDist === Infinity ? -1 : nearestDist,

            // Environment bounds (so the AI knows its limits)
            canvasWidth: CONFIG.CANVAS_WIDTH,
            groundY: CONFIG.GROUND_Y,
        };
    }

    /**
     * Get average survival time over recent rounds.
     * Useful for tracking learning progress.
     * @returns {number}
     */
    getAverageSurvival() {
        const recent = this.stats.recentSurvivals;
        if (recent.length === 0) return 0;
        return recent.reduce((a, b) => a + b, 0) / recent.length;
    }
}
