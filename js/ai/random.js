/**
 * random.js — Random Brain (Baseline AI)
 * 
 * PURPOSE:
 * This is the simplest possible AI — it picks a random action every time.
 * No learning, no strategy, no memory.
 * 
 * WHY DO WE NEED A BASELINE?
 * In AI research, you ALWAYS need a baseline to compare against:
 * - If your Q-Learning agent performs worse than random, there's a bug.
 * - If your DQN agent performs only slightly better than random, your
 *   reward function or state representation might need work.
 * - The gap between random and learned performance quantifies how much
 *   the AI has actually "learned."
 * 
 * EXPECTED BEHAVIOR:
 * - Enemy jitters randomly, occasionally dodging stones by luck
 * - Survival time should be low and inconsistent (high variance)
 * - No improvement over rounds (no learning)
 */

import { AIBrain } from './brain.js';
import { ACTION_LIST } from '../config.js';

export class RandomBrain extends AIBrain {
    constructor() {
        super('Random');
    }

    /**
     * Pick a completely random action from the action space.
     * 
     * Note: This is a uniform random policy — each action has equal
     * probability of being selected (1/6 ≈ 16.7% each).
     * 
     * @param {Object} gameState - ignored (random doesn't look at state)
     * @returns {string} A random action
     */
    decide(gameState) {
        const randomIndex = Math.floor(Math.random() * ACTION_LIST.length);
        return ACTION_LIST[randomIndex];
    }
}
