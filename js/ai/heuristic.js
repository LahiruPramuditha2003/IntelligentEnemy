/**
 * heuristic.js — Rule-Based (Heuristic) AI Brain
 * 
 * ============================================================================
 * AI CONCEPTS & ACADEMIC CONTEXT (For AI / Robotics / Game AI)
 * ============================================================================
 * 
 * 1. What is a Heuristic in AI?
 *    A heuristic is an engineered, domain-specific rule-of-thumb designed to solve
 *    a problem efficiently when exhaustive search or optimal planning is impractical.
 *    In game agents, heuristics typically take the form of:
 *    - Expert systems / Production rules (IF condition THEN action)
 *    - Potential fields (attraction to safety, repulsion from hazards)
 *    - Decision trees / Behavior trees
 * 
 * 2. Why build a Heuristic before Reinforcement Learning (RL)?
 *    - Strong Baseline: While RandomBrain gives the lower bound (chance behavior),
 *      HeuristicBrain gives a strong benchmark representing human-engineered logic.
 *    - Feature Engineering Discovery: Writing heuristic rules forces us to calculate
 *      geometric and physical relations (Time-to-Impact, danger zones, wall distance).
 *      These exact features will form the state vector for Q-Learning and Neural Networks!
 *    - Upper-Bound Comparison: Can a self-learning agent (Phase 3 & 4) independently
 *      discover evasion strategies that outperform our hand-crafted rules?
 * 
 * 3. The Flaw of Heuristics (Why We Need Learning!):
 *    - Brittle: The heuristic only handles scenarios anticipated by the engineer.
 *      If a player drops two stones in a trap configuration (one to flush the enemy,
 *      one where it will dodge to), the static rules cannot adapt or remember.
 *    - Rigid: Cannot exploit subtle environment dynamics without manual code rewrites.
 * 
 * ============================================================================
 * OUR HEURISTIC ALGORITHM:
 * ============================================================================
 * 1. Threat Filtering: Identify all stones whose landing X overlaps or nears the enemy.
 * 2. Time-to-Impact (TTI): Solve kinematic equation y(t) = y0 + vy*t + 0.5*g*t^2 to find
 *    the stone that will strike earliest.
 * 3. Directional Decision:
 *    - If stone is to the enemy's left, dodge right.
 *    - If stone is to the enemy's right, dodge left.
 *    - Wall proximity check: Do not dodge into a wall!
 *    - Speed control: Accelerate if TTI is critically small (< 0.6s).
 * 4. Center-Seeking: When no imminent threats exist, drift toward the center of the arena
 *    (center dominance provides maximum escape buffer in both directions).
 */

import { AIBrain } from './brain.js';
import { ACTIONS, CONFIG } from '../config.js';

export class HeuristicBrain extends AIBrain {
    constructor() {
        super('Heuristic');
    }

    /**
     * Compute Time-to-Impact (TTI) in seconds using kinematic quadratic formula.
     * y + vy*t + 0.5*g*t^2 = targetY
     * 0.5*g*t^2 + vy*t + (y - targetY) = 0
     * 
     * @param {number} stoneY 
     * @param {number} stoneVY 
     * @param {number} targetY 
     * @returns {number} Seconds until landing, or Infinity if unreachable
     */
    calculateTimeToImpact(stoneY, stoneVY, targetY) {
        const a = 0.5 * CONFIG.GRAVITY;
        const b = stoneVY;
        const c = stoneY - targetY;

        const discriminant = b * b - 4 * a * c;
        if (discriminant < 0) return Infinity;

        // Take positive root (time in future)
        const t = (-b + Math.sqrt(discriminant)) / (2 * a);
        return t >= 0 ? t : Infinity;
    }

    /**
     * Make an action decision based on domain rules.
     * 
     * @param {Object} gameState 
     * @returns {string} Action from ACTIONS
     */
    decide(gameState) {
        const {
            enemyX,
            enemyY,
            enemySpeed,
            enemyIsJumping,
            stones,
            canvasWidth,
            groundY
        } = gameState;

        const enemyWidth = CONFIG.ENEMY_WIDTH;
        const enemyCenterX = enemyX + enemyWidth / 2;
        const safetyMargin = CONFIG.STONE_RADIUS + 25; // Horizontal safety buffer

        // If no active stones, return to center for safety
        if (!stones || stones.length === 0) {
            return this.seekCenter(enemyCenterX, canvasWidth);
        }

        // 1. Analyze threats: Find stones falling in or near our horizontal corridor
        let mostCriticalThreat = null;
        let minTimeToImpact = Infinity;

        for (const stone of stones) {
            // Horizontal distance from stone to enemy center
            const xDist = Math.abs(stone.x - enemyCenterX);

            // Is the stone falling toward our zone?
            const isHorizontalThreat = xDist < (enemyWidth / 2 + safetyMargin);
            const isAboveEnemy = stone.y < enemyY + CONFIG.ENEMY_HEIGHT;

            if (isHorizontalThreat && isAboveEnemy) {
                const tti = this.calculateTimeToImpact(stone.y, stone.vy, groundY);
                if (tti < minTimeToImpact) {
                    minTimeToImpact = tti;
                    mostCriticalThreat = stone;
                }
            }
        }

        // 2. If an imminent threat exists, execute evasion
        if (mostCriticalThreat && minTimeToImpact < 1.5) {
            const stoneX = mostCriticalThreat.x;
            const distToLeftWall = enemyX;
            const distToRightWall = canvasWidth - (enemyX + enemyWidth);

            // Ensure we are moving fast enough to dodge in time
            if (minTimeToImpact < 0.8 && enemySpeed < CONFIG.ENEMY_MAX_SPEED) {
                return ACTIONS.SPEED_UP;
            }

            // Decide dodge direction away from stone
            const stoneIsRight = stoneX >= enemyCenterX;
            const stoneIsLeft = stoneX < enemyCenterX;

            // Check if trapped against a wall
            const nearLeftWall = distToLeftWall < 40;
            const nearRightWall = distToRightWall < 40;

            if (stoneIsRight) {
                // Stone is on the right -> want to move left
                if (!nearLeftWall) {
                    return ACTIONS.MOVE_LEFT;
                } else {
                    // Cornered against left wall! Emergency jump or push right under stone
                    if (!enemyIsJumping && minTimeToImpact < 0.5) {
                        return ACTIONS.JUMP;
                    }
                    return ACTIONS.MOVE_RIGHT;
                }
            } else if (stoneIsLeft) {
                // Stone is on the left -> want to move right
                if (!nearRightWall) {
                    return ACTIONS.MOVE_RIGHT;
                } else {
                    // Cornered against right wall!
                    if (!enemyIsJumping && minTimeToImpact < 0.5) {
                        return ACTIONS.JUMP;
                    }
                    return ACTIONS.MOVE_LEFT;
                }
            }
        }

        // 3. No immediate threat overhead:
        // If speed is abnormally high, slow down slightly to retain control,
        // and drift toward center.
        if (enemySpeed > CONFIG.ENEMY_BASE_SPEED + 50) {
            return ACTIONS.SLOW_DOWN;
        }

        return this.seekCenter(enemyCenterX, canvasWidth);
    }

    /**
     * Subtle drift towards center of canvas to maintain maneuverability.
     */
    seekCenter(enemyCenterX, canvasWidth) {
        const center = canvasWidth / 2;
        const tolerance = 40;

        if (enemyCenterX < center - tolerance) {
            return ACTIONS.MOVE_RIGHT;
        } else if (enemyCenterX > center + tolerance) {
            return ACTIONS.MOVE_LEFT;
        }
        return ACTIONS.STAY;
    }
}
