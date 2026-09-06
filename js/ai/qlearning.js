/**
 * qlearning.js — Tabular Q-Learning (Reinforcement Learning) Brain
 * 
 * ============================================================================
 * REINFORCEMENT LEARNING (RL) THEORETICAL FOUNDATIONS
 * ============================================================================
 * 
 * 1. The Reinforcement Learning Problem:
 *    Unlike Supervised Learning (where correct labels are given) or Unsupervised
 *    Learning (discovering hidden patterns), an RL agent learns by TRIAL AND ERROR
 *    through interaction with an environment to maximize cumulative reward.
 * 
 *    Formally modeled as a Markov Decision Process (MDP):
 *    - S: State space (what the agent perceives)
 *    - A: Action space (what the agent can do)
 *    - P(s' | s, a): Transition dynamics (physics / game engine)
 *    - R(s, a): Reward function (reinforcement signal)
 *    - γ (gamma): Discount factor for future rewards (0 <= γ < 1)
 * 
 * 2. What is Q-Learning? (Watkins, 1989):
 *    Q-Learning is a model-free, off-policy Temporal Difference (TD) algorithm.
 *    The "Q" stands for "Quality" of taking action `a` in state `s`.
 * 
 *    Q*(s, a) represents the expected sum of discounted future rewards:
 *      Q*(s, a) = E [ R_{t+1} + γ * max_{a'} Q*(S_{t+1}, a') | S_t = s, A_t = a ]
 * 
 * 3. The Bellman Optimality Equation & TD Update Rule:
 *    At each step, when transitioning from state s to s' with reward r:
 * 
 *      TD Target  = r + γ * max_{a'} Q(s', a')   (for non-terminal state)
 *      TD Target  = r                             (for terminal state / death)
 *      TD Error δ = TD Target - Q(s, a)
 * 
 *      New Q(s, a) = Q(s, a) + α * δ
 *                  = Q(s, a) + α * [ r + γ * max_{a'} Q(s', a') - Q(s, a) ]
 * 
 *    where:
 *      α (alpha) = Learning Rate (how much new information overrides old)
 *      γ (gamma) = Discount Factor (how much future rewards matter vs immediate)
 * 
 * 4. Exploration vs. Exploitation (ε-greedy strategy):
 *    - If the agent only exploits (picks highest Q), it may never discover better moves.
 *    - If it only explores (random moves), it never capitalizes on what it learned.
 *    - Solution: ε-greedy!
 *        With probability ε: pick a RANDOM action (explore)
 *        With probability (1 - ε): pick argmax_a Q(s, a) (exploit)
 *    - Over time, ε decays from 1.0 down to a small floor (e.g. 0.05).
 * 
 * 5. State Discretization (Tackling the Curse of Dimensionality):
 *    The game's continuous coordinates (x, y, vy) are infinite. A lookup table cannot
 *    generalize over infinite continuous floats without discretization.
 *    We bin continuous metrics into semantic discrete buckets:
 *      - Relative horizontal threat zone: FAR_LEFT, CLOSE_LEFT, OVERHEAD, CLOSE_RIGHT, FAR_RIGHT, NONE (6)
 *      - Threat altitude: HIGH, MID, LOW, NONE (4)
 *      - Wall proximity: NEAR_LEFT, CENTER, NEAR_RIGHT (3)
 *      - Jump state: GROUND, AIR (2)
 *    Total State Space: 6 * 4 * 3 * 2 = 144 discrete states.
 *    Total Q-table size: 144 * 6 actions = 864 values. Fast to converge!
 */

import { AIBrain } from './brain.js';
import { ACTIONS, ACTION_LIST, CONFIG } from '../config.js';

export class QLearningBrain extends AIBrain {
    /**
     * @param {Object} options - Hyperparameters
     */
    constructor(options = {}) {
        super('Q-Learning');

        // === HYPERPARAMETERS ===
        this.alpha = options.alpha ?? 0.2;         // Learning rate (α)
        this.gamma = options.gamma ?? 0.9;         // Discount factor (γ)
        this.epsilon = options.epsilon ?? 0.7;     // Initial exploration rate (ε)
        this.epsilonMin = options.epsilonMin ?? 0.05; // Minimum exploration rate
        this.epsilonDecay = options.epsilonDecay ?? 0.98; // Decay per round

        // === Q-TABLE ===
        // Maps stateString -> { [action]: number }
        this.qTable = {};

        // === EPISODIC TRACKING ===
        this.prevState = null;
        this.prevAction = null;
        this.totalUpdates = 0;
        this.totalRoundsTrained = 0;

        // Try restoring learned knowledge from localStorage
        this.loadFromStorage();
    }

    /**
     * Discretize continuous game state into a compact discrete string key.
     * 
     * @param {Object} gameState 
     * @returns {string} e.g. "OVERHEAD|LOW|CENTER|GROUND"
     */
    discretizeState(gameState) {
        const {
            enemyX,
            enemyWidth = CONFIG.ENEMY_WIDTH,
            enemyIsJumping,
            stones,
            canvasWidth = CONFIG.CANVAS_WIDTH,
            groundY = CONFIG.GROUND_Y,
        } = gameState;

        const enemyCenterX = enemyX + enemyWidth / 2;

        // 1. Identify the most critical active stone
        let criticalStone = null;
        let lowestY = -1;

        if (stones && stones.length > 0) {
            for (const stone of stones) {
                // Focus on stones above ground level
                if (stone.y < groundY && stone.y > lowestY) {
                    lowestY = stone.y;
                    criticalStone = stone;
                }
            }
        }

        // 2. Discretize Horizontal Threat Zone
        let threatZone = 'NONE';
        let altitudeZone = 'NONE';

        if (criticalStone) {
            const dx = criticalStone.x - enemyCenterX;

            if (dx < -60) {
                threatZone = 'FAR_LEFT';
            } else if (dx < -15) {
                threatZone = 'CLOSE_LEFT';
            } else if (dx <= 15) {
                threatZone = 'OVERHEAD'; // Highest direct danger
            } else if (dx <= 60) {
                threatZone = 'CLOSE_RIGHT';
            } else {
                threatZone = 'FAR_RIGHT';
            }

            // 3. Discretize Altitude / Time-to-impact
            if (criticalStone.y < 220) {
                altitudeZone = 'HIGH';
            } else if (criticalStone.y < 420) {
                altitudeZone = 'MID';
            } else {
                altitudeZone = 'LOW'; // Imminent ground impact
            }
        }

        // 4. Discretize Wall Proximity
        let wallZone = 'CENTER';
        if (enemyX < 50) {
            wallZone = 'NEAR_LEFT';
        } else if (enemyX + enemyWidth > canvasWidth - 50) {
            wallZone = 'NEAR_RIGHT';
        }

        // 5. Discretize Jumping State
        const jumpZone = enemyIsJumping ? 'AIR' : 'GROUND';

        return `${threatZone}|${altitudeZone}|${wallZone}|${jumpZone}`;
    }

    /**
     * Get or initialize Q-values for a given discrete state.
     * 
     * Optimistic vs Zero Initialization:
     * - Zero init: all Q = 0
     * - Optimistic init (e.g. Q = 10): encourages early exploration of all actions
     * We use zero initialization here with explicit ε-greedy exploration.
     * 
     * @param {string} stateKey 
     * @returns {Object} { [action]: qValue }
     */
    getQValues(stateKey) {
        if (!this.qTable[stateKey]) {
            this.qTable[stateKey] = {};
            for (const action of ACTION_LIST) {
                this.qTable[stateKey][action] = 0;
            }
        }
        return this.qTable[stateKey];
    }

    /**
     * Find the action with the highest Q-value in the given state (Greedy Choice).
     * Ties are broken randomly to avoid deterministic lock-in.
     * 
     * @param {string} stateKey 
     * @returns {string} Best action
     */
    getBestAction(stateKey) {
        const qValues = this.getQValues(stateKey);
        let maxQ = -Infinity;
        let bestActions = [];

        for (const action of ACTION_LIST) {
            const q = qValues[action];
            if (q > maxQ) {
                maxQ = q;
                bestActions = [action];
            } else if (q === maxQ) {
                bestActions.push(action);
            }
        }

        // Random tie-break
        const idx = Math.floor(Math.random() * bestActions.length);
        return bestActions[idx];
    }

    /**
     * Execute one TD step update:
     * Q(s, a) ← Q(s, a) + α * [r + γ * max_{a'} Q(s', a') - Q(s, a)]
     */
    updateQ(state, action, reward, nextState, isTerminal = false) {
        const qValues = this.getQValues(state);
        const currentQ = qValues[action];

        let target = reward;
        if (!isTerminal && nextState) {
            const nextQValues = this.getQValues(nextState);
            let maxNextQ = -Infinity;
            for (const a of ACTION_LIST) {
                if (nextQValues[a] > maxNextQ) {
                    maxNextQ = nextQValues[a];
                }
            }
            target = reward + this.gamma * maxNextQ;
        }

        const tdError = target - currentQ;
        qValues[action] = currentQ + this.alpha * tdError;
        this.totalUpdates++;
    }

    /**
     * Core Decision Method:
     * 1. Perform TD-update for previous transition (s_prev, a_prev) -> s_curr with survival reward.
     * 2. Choose next action using ε-greedy policy.
     * 
     * @param {Object} gameState 
     * @returns {string} Chosen action
     */
    decide(gameState) {
        const currentStateKey = this.discretizeState(gameState);

        // 1. TD-update on previous step (Living reward: survived another decision interval!)
        if (this.prevState !== null && this.prevAction !== null) {
            // Reward shaping:
            // Base survival reward: +1
            // Overhead evasion bonus: if previous state had an overhead stone and now it does not, +3!
            let stepReward = 1.0;
            if (this.prevState.includes('OVERHEAD') && !currentStateKey.includes('OVERHEAD')) {
                stepReward += 3.0; // Rewarding active dodging away from danger
            }

            this.updateQ(this.prevState, this.prevAction, stepReward, currentStateKey, false);
        }

        // 2. Action Selection: ε-greedy
        let selectedAction;
        if (Math.random() < this.epsilon) {
            // Exploration: choose uniform random action
            const randIdx = Math.floor(Math.random() * ACTION_LIST.length);
            selectedAction = ACTION_LIST[randIdx];
        } else {
            // Exploitation: choose greedy action with highest Q
            selectedAction = this.getBestAction(currentStateKey);
        }

        // Store current for the next transition update
        this.prevState = currentStateKey;
        this.prevAction = selectedAction;

        return selectedAction;
    }

    /**
     * Called when enemy is hit by a stone (Terminal State).
     * Strong negative reinforcement signal (-100).
     */
    onDeath(gameState) {
        if (this.prevState !== null && this.prevAction !== null) {
            // Death penalty: -100
            const deathPenalty = -100.0;
            this.updateQ(this.prevState, this.prevAction, deathPenalty, null, true);
        }

        this.prevState = null;
        this.prevAction = null;
        this.totalRoundsTrained++;

        // Decay exploration rate ε
        this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);

        // Auto-persist knowledge to localStorage
        this.saveToStorage();
    }

    /**
     * Prepare for new round.
     */
    reset() {
        this.prevState = null;
        this.prevAction = null;
    }

    /**
     * Reset all learning (clean slate).
     */
    clearKnowledge() {
        this.qTable = {};
        this.totalUpdates = 0;
        this.totalRoundsTrained = 0;
        this.epsilon = 0.7;
        this.prevState = null;
        this.prevAction = null;
        try {
            localStorage.removeItem('stone_drop_qtable');
            localStorage.removeItem('stone_drop_qmeta');
        } catch (e) {
            // Ignore storage errors in restricted contexts
        }
    }

    /**
     * Save Q-table to browser localStorage.
     */
    saveToStorage() {
        try {
            localStorage.setItem('stone_drop_qtable', JSON.stringify(this.qTable));
            localStorage.setItem('stone_drop_qmeta', JSON.stringify({
                epsilon: this.epsilon,
                totalUpdates: this.totalUpdates,
                totalRoundsTrained: this.totalRoundsTrained,
            }));
        } catch (e) {
            // Storage quota or restriction
        }
    }

    /**
     * Load Q-table from browser localStorage.
     */
    loadFromStorage() {
        try {
            const rawTable = localStorage.getItem('stone_drop_qtable');
            const rawMeta = localStorage.getItem('stone_drop_qmeta');
            if (rawTable) {
                this.qTable = JSON.parse(rawTable);
            }
            if (rawMeta) {
                const meta = JSON.parse(rawMeta);
                this.epsilon = meta.epsilon ?? this.epsilon;
                this.totalUpdates = meta.totalUpdates ?? 0;
                this.totalRoundsTrained = meta.totalRoundsTrained ?? 0;
            }
        } catch (e) {
            // Ignore corrupted storage
        }
    }

    /**
     * Metrics for HUD/UI display.
     */
    getStats() {
        const stateCount = Object.keys(this.qTable).length;
        return {
            epsilon: this.epsilon.toFixed(2),
            statesDiscovered: stateCount,
            totalUpdates: this.totalUpdates,
            roundsTrained: this.totalRoundsTrained,
        };
    }
}
