/**
 * dqn.js — Deep Q-Network (DQN) Brain
 * 
 * ============================================================================
 * THEORETICAL FOUNDATIONS (Deep Q-Networks — Mnih et al., Nature 2015)
 * ============================================================================
 * 
 * In Tabular Q-Learning, the agent maintains a lookup table. But in real-world AI,
 * state spaces are continuous and high-dimensional. DQN solves this using Deep Neural
 * Networks as function approximators, enhanced by TWO revolutionary breakthroughs:
 * 
 * 1. Experience Replay (Lin, 1992; Mnih et al., 2015):
 *    - In standard RL, the agent learns online from sequential steps (s_t, a_t, r_t, s_{t+1}).
 *    - Problem: Consecutive states are strongly correlated, violating the Independent
 *      and Identically Distributed (i.i.d.) assumption of gradient descent.
 *    - Solution: Store experiences in a circular buffer D = {e_1, e_2, ...}.
 *      During training, sample uniform random mini-batches from D to break temporal correlation!
 * 
 * 2. Target Network (Mnih et al., 2015):
 *    - In standard Q-learning with function approximation:
 *        Loss = ( r + γ * max_{a'} Q(s', a'; θ) - Q(s, a; θ) )^2
 *    - Problem: The target uses the same parameter vector θ that is being updated!
 *      This creates a moving target (like a dog chasing its own tail), causing oscillations
 *      and divergence.
 *    - Solution: Use a separate Target Network with frozen weights θ⁻:
 *        Target y = r + γ * max_{a'} Q(s', a'; θ⁻)
 *      The target weights θ⁻ are updated only periodically (every C steps).
 * 
 * 3. Continuous State Vector (7 normalized features):
 *    [0] enemyX_norm   : horizontal position in [0, 1]
 *    [1] enemyVX_norm  : horizontal velocity in [-1, 1]
 *    [2] enemyJumping  : 1 if airborne, 0 if grounded
 *    [3] hasThreat     : 1 if stone active, 0 otherwise
 *    [4] relStoneX     : relative horizontal offset to nearest stone in [-1, 1]
 *    [5] relStoneY     : stone vertical altitude in [0, 1]
 *    [6] stoneVY_norm  : stone vertical velocity in [0, 1]
 */

import { AIBrain } from './brain.js';
import { ACTIONS, ACTION_LIST, CONFIG } from '../config.js';
import { NeuralNetwork } from './neural_net.js';

export class DQNBrain extends AIBrain {
    /**
     * @param {Object} options - Hyperparameters
     */
    constructor(options = {}) {
        super('DQN (Neural Net)');

        // === HYPERPARAMETERS ===
        this.gamma = options.gamma ?? 0.95;             // Discount factor
        this.learningRate = options.learningRate ?? 0.005; // Adam learning rate
        this.epsilon = options.epsilon ?? 0.8;         // Initial exploration rate
        this.epsilonMin = options.epsilonMin ?? 0.05;   // Minimum exploration rate
        this.epsilonDecay = options.epsilonDecay ?? 0.97; // Decay per episode
        this.batchSize = options.batchSize ?? 16;       // Mini-batch size
        this.targetUpdateFreq = options.targetUpdateFreq ?? 30; // Steps between target net sync
        this.replayCapacity = options.replayCapacity ?? 1000;   // Buffer size

        // === NEURAL NETWORKS ===
        // Input: 7 continuous features -> Hidden: 24 -> Hidden: 24 -> Output: 6 Q-values
        this.inputDim = 7;
        this.outputDim = ACTION_LIST.length;
        this.policyNet = new NeuralNetwork(this.inputDim, [24, 24], this.outputDim);
        this.targetNet = new NeuralNetwork(this.inputDim, [24, 24], this.outputDim);
        this.targetNet.copyWeightsFrom(this.policyNet);

        // === EXPERIENCE REPLAY BUFFER ===
        this.replayBuffer = [];
        this.bufferPointer = 0;

        // === TRACKING STATE ===
        this.prevState = null;
        this.prevActionIndex = null;
        this.stepCounter = 0;
        this.totalUpdates = 0;
        this.totalRoundsTrained = 0;
        this.recentLosses = [];

        // Restore pre-trained weights if available
        this.loadFromStorage();
    }

    /**
     * Extract normalized continuous state vector [7 features].
     * @param {Object} gameState 
     * @returns {Float32Array}
     */
    extractStateVector(gameState) {
        const {
            enemyX,
            enemyVX,
            enemyWidth = CONFIG.ENEMY_WIDTH,
            enemyIsJumping,
            stones,
            canvasWidth = CONFIG.CANVAS_WIDTH,
            groundY = CONFIG.GROUND_Y,
        } = gameState;

        const enemyCenterX = enemyX + enemyWidth / 2;
        const maxSpeed = CONFIG.ENEMY_MAX_SPEED;

        // Find nearest active stone
        let nearestStone = null;
        let lowestY = -1;
        if (stones && stones.length > 0) {
            for (const s of stones) {
                if (s.y < groundY && s.y > lowestY) {
                    lowestY = s.y;
                    nearestStone = s;
                }
            }
        }

        const state = new Float32Array(this.inputDim);
        // [0] Enemy X normalized [0, 1]
        state[0] = Math.max(0, Math.min(1, enemyCenterX / canvasWidth));
        // [1] Enemy VX normalized [-1, 1]
        state[1] = Math.max(-1, Math.min(1, enemyVX / maxSpeed));
        // [2] Enemy Jumping flag
        state[2] = enemyIsJumping ? 1.0 : 0.0;

        if (nearestStone) {
            // [3] Has threat flag
            state[3] = 1.0;
            // [4] Relative stone X offset [-1, 1]
            const dx = nearestStone.x - enemyCenterX;
            state[4] = Math.max(-1, Math.min(1, dx / (canvasWidth / 2)));
            // [5] Stone Y normalized [0, 1]
            state[5] = Math.max(0, Math.min(1, nearestStone.y / groundY));
            // [6] Stone VY normalized [0, 1]
            state[6] = Math.max(0, Math.min(1, nearestStone.vy / 1000));
        } else {
            state[3] = 0.0;
            state[4] = 0.0;
            state[5] = 0.0;
            state[6] = 0.0;
        }

        return state;
    }

    /**
     * Store an experience transition in the replay buffer.
     */
    addExperience(state, actionIndex, reward, nextState, done) {
        const transition = { state, actionIndex, reward, nextState, done };
        if (this.replayBuffer.length < this.replayCapacity) {
            this.replayBuffer.push(transition);
        } else {
            this.replayBuffer[this.bufferPointer] = transition;
            this.bufferPointer = (this.bufferPointer + 1) % this.replayCapacity;
        }
    }

    /**
     * Train the policy network on a random mini-batch from the replay buffer.
     */
    trainMiniBatch() {
        if (this.replayBuffer.length < this.batchSize) return;

        let totalLoss = 0;

        // Sample uniform random mini-batch
        for (let i = 0; i < this.batchSize; i++) {
            const randIdx = Math.floor(Math.random() * this.replayBuffer.length);
            const { state, actionIndex, reward, nextState, done } = this.replayBuffer[randIdx];

            let target = reward;
            if (!done && nextState) {
                // Double Q-Learning or standard DQN Target Network evaluation
                // Target Q = reward + γ * max_{a'} Q_target(nextState, a')
                const targetQValues = this.targetNet.predict(nextState);
                let maxNextQ = -Infinity;
                for (let a = 0; a < targetQValues.length; a++) {
                    if (targetQValues[a] > maxNextQ) {
                        maxNextQ = targetQValues[a];
                    }
                }
                target = reward + this.gamma * maxNextQ;
            }

            // Compute loss & accumulate gradients
            const loss = this.policyNet.backwardSingle(state, actionIndex, target);
            totalLoss += loss;
        }

        // Apply Adam update
        this.policyNet.applyGradients(this.learningRate, this.batchSize);
        this.totalUpdates++;

        // Track rolling loss
        this.recentLosses.push(totalLoss / this.batchSize);
        if (this.recentLosses.length > 30) this.recentLosses.shift();

        // Target network periodic synchronization
        this.stepCounter++;
        if (this.stepCounter % this.targetUpdateFreq === 0) {
            this.targetNet.copyWeightsFrom(this.policyNet);
        }
    }

    /**
     * Main action decision method using ε-greedy policy over neural network outputs.
     * @param {Object} gameState 
     * @returns {string} Action name from ACTIONS
     */
    decide(gameState) {
        const currentStateVector = this.extractStateVector(gameState);

        // 1. Process previous transition
        if (this.prevState !== null && this.prevActionIndex !== null) {
            // Reward shaping:
            // Survival reward: +1
            // Overhead evasion bonus: +3 if dodging an overhead stone
            let stepReward = 1.0;
            const prevDx = Math.abs(this.prevState[4]); // relative horizontal offset
            const currDx = Math.abs(currentStateVector[4]);
            if (prevDx < 0.1 && currDx >= 0.1 && currentStateVector[3] === 1.0) {
                stepReward += 3.0; // Dodged out of direct path
            }

            this.addExperience(this.prevState, this.prevActionIndex, stepReward, currentStateVector, false);
            this.trainMiniBatch();
        }

        // 2. Action selection: ε-greedy
        let chosenActionIndex;
        if (Math.random() < this.epsilon) {
            // Explore
            chosenActionIndex = Math.floor(Math.random() * ACTION_LIST.length);
        } else {
            // Exploit: forward pass through Neural Network
            const qValues = this.policyNet.predict(currentStateVector);
            let maxQ = -Infinity;
            chosenActionIndex = 0;
            for (let i = 0; i < qValues.length; i++) {
                if (qValues[i] > maxQ) {
                    maxQ = qValues[i];
                    chosenActionIndex = i;
                }
            }
        }

        this.prevState = currentStateVector;
        this.prevActionIndex = chosenActionIndex;

        return ACTION_LIST[chosenActionIndex];
    }

    /**
     * Called when enemy is hit by a stone (Terminal State).
     */
    onDeath(gameState) {
        if (this.prevState !== null && this.prevActionIndex !== null) {
            // Terminal penalty: -100
            const deathPenalty = -100.0;
            this.addExperience(this.prevState, this.prevActionIndex, deathPenalty, null, true);
            // Run extra training iterations on terminal transition
            this.trainMiniBatch();
            this.trainMiniBatch();
        }

        this.prevState = null;
        this.prevActionIndex = null;
        this.totalRoundsTrained++;

        // Decay exploration rate
        this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);

        // Sync target network on death as well
        this.targetNet.copyWeightsFrom(this.policyNet);

        // Auto-save
        this.saveToStorage();
    }

    /**
     * Reset episodic state for a new round.
     */
    reset() {
        this.prevState = null;
        this.prevActionIndex = null;
    }

    /**
     * Clear all learned neural weights and replay memory.
     */
    clearKnowledge() {
        this.policyNet = new NeuralNetwork(this.inputDim, [24, 24], this.outputDim);
        this.targetNet = new NeuralNetwork(this.inputDim, [24, 24], this.outputDim);
        this.targetNet.copyWeightsFrom(this.policyNet);
        this.replayBuffer = [];
        this.bufferPointer = 0;
        this.totalUpdates = 0;
        this.totalRoundsTrained = 0;
        this.epsilon = 0.8;
        this.prevState = null;
        this.prevActionIndex = null;
        this.recentLosses = [];
        try {
            localStorage.removeItem('stone_drop_dqn_weights');
            localStorage.removeItem('stone_drop_dqn_meta');
        } catch (e) {
            // Ignore storage issues
        }
    }

    /**
     * Save network weights to localStorage.
     */
    saveToStorage() {
        try {
            const weightsJSON = JSON.stringify(this.policyNet.toJSON());
            const metaJSON = JSON.stringify({
                epsilon: this.epsilon,
                totalUpdates: this.totalUpdates,
                totalRoundsTrained: this.totalRoundsTrained,
            });
            localStorage.setItem('stone_drop_dqn_weights', weightsJSON);
            localStorage.setItem('stone_drop_dqn_meta', metaJSON);
        } catch (e) {
            // Storage quota
        }
    }

    /**
     * Load network weights from localStorage.
     */
    loadFromStorage() {
        try {
            const rawWeights = localStorage.getItem('stone_drop_dqn_weights');
            const rawMeta = localStorage.getItem('stone_drop_dqn_meta');
            if (rawWeights) {
                this.policyNet.fromJSON(JSON.parse(rawWeights));
                this.targetNet.copyWeightsFrom(this.policyNet);
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
        const avgLoss = this.recentLosses.length > 0 
            ? (this.recentLosses.reduce((a, b) => a + b, 0) / this.recentLosses.length).toFixed(4)
            : '0.0000';

        return {
            epsilon: this.epsilon.toFixed(2),
            statesDiscovered: `Replay: ${this.replayBuffer.length}`,
            roundsTrained: this.totalRoundsTrained,
            totalUpdates: this.totalUpdates,
            loss: avgLoss,
        };
    }
}
