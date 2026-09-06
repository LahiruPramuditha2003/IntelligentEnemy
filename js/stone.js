/**
 * stone.js — Stone entity and StoneManager.
 * 
 * Stones are the player's "weapon" — they spawn at the top of the screen
 * and fall under gravity. If one hits the enemy, the round ends.
 * 
 * From the AI's perspective, stones are the "environment hazards" that
 * the agent must learn to avoid. The stone positions and velocities
 * form part of the STATE that the AI observes.
 */

import { CONFIG } from './config.js';

export class Stone {
    /**
     * Create a new stone at the top of the screen.
     * @param {number} x - Horizontal position (where the player clicked)
     */
    constructor(x) {
        this.x = x;
        this.y = 0;                         // Start at the very top
        this.radius = CONFIG.STONE_RADIUS;
        this.vy = 0;                        // Starts stationary, gravity accelerates it
        this.active = true;                 // false = should be cleaned up
    }

    /**
     * Update stone physics for one frame.
     * Simple free-fall: velocity increases by gravity each frame,
     * position increases by velocity each frame.
     * 
     * v(t+dt) = v(t) + g * dt       (velocity update)
     * y(t+dt) = y(t) + v(t+dt) * dt (position update, semi-implicit Euler)
     * 
     * @param {number} dt - Delta time in seconds
     */
    update(dt) {
        if (!this.active) return;

        // Accelerate due to gravity
        this.vy += CONFIG.GRAVITY * dt;

        // Fall
        this.y += this.vy * dt;

        // Deactivate if past the ground (missed the enemy)
        if (this.y - this.radius >= CONFIG.GROUND_Y) {
            this.active = false;
        }
    }
}

/**
 * StoneManager — manages the lifecycle of all stones.
 * 
 * Responsibilities:
 * - Spawn new stones (with cooldown to prevent spam)
 * - Update all active stones
 * - Clean up deactivated stones (garbage collection)
 * - Provide active stone list for collision and rendering
 */
export class StoneManager {
    constructor() {
        this.stones = [];
        this.lastSpawnTime = 0;
    }

    /**
     * Attempt to spawn a new stone.
     * Respects the cooldown timer to prevent the player from
     * overwhelming the AI with too many stones at once.
     * 
     * @param {number} x - X position to spawn at
     * @param {number} currentTime - Current timestamp (ms)
     * @returns {boolean} Whether the stone was successfully spawned
     */
    spawn(x, currentTime) {
        if (currentTime - this.lastSpawnTime < CONFIG.STONE_SPAWN_COOLDOWN) {
            return false;   // Cooldown not elapsed
        }

        this.stones.push(new Stone(x));
        this.lastSpawnTime = currentTime;
        return true;
    }

    /**
     * Update all stones and remove inactive ones.
     * @param {number} dt - Delta time in seconds
     * @param {Function} [onGroundHit] - Callback (x, y) when a stone strikes the ground
     */
    update(dt, onGroundHit = null) {
        for (const stone of this.stones) {
            const wasActive = stone.active;
            stone.update(dt);
            if (wasActive && !stone.active && onGroundHit) {
                onGroundHit(stone.x, CONFIG.GROUND_Y);
            }
        }

        // Garbage collect: remove stones that have hit the ground or been deactivated
        this.stones = this.stones.filter(s => s.active);
    }

    /**
     * Get all currently active (in-flight) stones.
     * @returns {Stone[]}
     */
    getActiveStones() {
        return this.stones.filter(s => s.active);
    }

    /**
     * Reset all stones (for new round).
     */
    reset() {
        this.stones = [];
        this.lastSpawnTime = 0;
    }
}
