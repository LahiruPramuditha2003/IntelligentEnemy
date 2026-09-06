/**
 * enemy.js — The Enemy entity.
 * 
 * The enemy is the AI-controlled agent in our game. It exists on the ground
 * and can perform discrete actions to avoid falling stones.
 * 
 * KEY CONCEPT: Action Space
 * The enemy can perform 6 discrete actions (defined in config.js).
 * This fixed action set is fundamental to Reinforcement Learning:
 * - The AI doesn't choose raw velocities (continuous control)
 * - It picks from a menu of predefined behaviors
 * - This makes Q-tables and neural network outputs tractable
 * 
 * The enemy maintains its own physics state (position, velocity) and
 * applies actions through the applyAction() method. The game loop calls
 * update() each frame to advance the physics simulation.
 */

import { CONFIG, ACTIONS } from './config.js';

export class Enemy {
    constructor() {
        this.width = CONFIG.ENEMY_WIDTH;
        this.height = CONFIG.ENEMY_HEIGHT;
        this.reset();
    }

    /**
     * Reset the enemy to its starting state.
     * Called at the beginning of each round.
     */
    reset() {
        // Start at center of the ground
        this.x = (CONFIG.CANVAS_WIDTH / 2) - (this.width / 2);
        this.y = CONFIG.GROUND_Y - this.height;

        // Velocity
        this.vx = 0;   // horizontal velocity (px/s)
        this.vy = 0;   // vertical velocity (px/s)

        // Movement parameters
        this.speed = CONFIG.ENEMY_BASE_SPEED;

        // State flags
        this.isJumping = false;
        this.isAlive = true;
    }

    /**
     * Apply a discrete action to the enemy.
     * This is called by the AI brain's decision — the action is translated
     * into physics changes here.
     * 
     * @param {string} action - One of the ACTIONS values
     */
    applyAction(action) {
        if (!this.isAlive) return;

        switch (action) {
            case ACTIONS.MOVE_LEFT:
                this.vx = -this.speed;
                break;

            case ACTIONS.MOVE_RIGHT:
                this.vx = this.speed;
                break;

            case ACTIONS.JUMP:
                // Can only jump if on the ground (prevents infinite jumps)
                if (!this.isJumping) {
                    this.vy = CONFIG.ENEMY_JUMP_FORCE;  // negative = upward
                    this.isJumping = true;
                }
                break;

            case ACTIONS.SPEED_UP:
                this.speed = Math.min(
                    this.speed + CONFIG.ENEMY_SPEED_STEP,
                    CONFIG.ENEMY_MAX_SPEED
                );
                break;

            case ACTIONS.SLOW_DOWN:
                this.speed = Math.max(
                    this.speed - CONFIG.ENEMY_SPEED_STEP,
                    CONFIG.ENEMY_MIN_SPEED
                );
                break;

            case ACTIONS.STAY:
                this.vx = 0;
                break;
        }
    }

    /**
     * Update enemy physics for one frame.
     * 
     * Physics model:
     * - Gravity pulls the enemy down (for jumping)
     * - Position updates from velocity (Euler integration)
     * - Ground and wall collisions keep enemy in bounds
     * 
     * @param {number} dt - Delta time in seconds
     */
    update(dt) {
        if (!this.isAlive) return;

        // Apply gravity (only affects vertical — horizontal is action-driven)
        this.vy += CONFIG.GRAVITY * dt;

        // Euler integration: position += velocity * time
        this.x += this.vx * dt;
        this.y += this.vy * dt;

        // --- Boundary constraints ---

        // Ground collision: snap to ground, stop falling
        if (this.y + this.height >= CONFIG.GROUND_Y) {
            this.y = CONFIG.GROUND_Y - this.height;
            this.vy = 0;
            this.isJumping = false;
        }

        // Left wall
        if (this.x < 0) {
            this.x = 0;
            this.vx = 0;
        }

        // Right wall
        if (this.x + this.width > CONFIG.CANVAS_WIDTH) {
            this.x = CONFIG.CANVAS_WIDTH - this.width;
            this.vx = 0;
        }
    }

    /**
     * Kill the enemy (called on stone collision).
     */
    die() {
        this.isAlive = false;
        this.vx = 0;
        this.vy = 0;
    }

    /**
     * Get the enemy's axis-aligned bounding box.
     * Used by the collision detection system.
     * 
     * @returns {{x: number, y: number, width: number, height: number}}
     */
    getBounds() {
        return {
            x: this.x,
            y: this.y,
            width: this.width,
            height: this.height,
        };
    }
}
