/**
 * config.js — Central configuration for the Stone Drop game.
 * 
 * ALL tunable parameters live here. This is critical for:
 * 1. Quick experimentation (change one number, see the effect)
 * 2. AI hyperparameter tuning (learning rate, discount factor, etc. will go here)
 * 3. Difficulty balancing (enemy speed, gravity, cooldowns)
 */

export const CONFIG = {
    // === Canvas ===
    CANVAS_WIDTH: 800,
    CANVAS_HEIGHT: 600,
    GROUND_Y: 550,              // Y-coordinate of the ground line

    // === Enemy ===
    ENEMY_WIDTH: 30,
    ENEMY_HEIGHT: 40,
    ENEMY_BASE_SPEED: 200,      // px/s — starting horizontal speed
    ENEMY_MIN_SPEED: 100,       // px/s — minimum after SLOW_DOWN
    ENEMY_MAX_SPEED: 400,       // px/s — maximum after SPEED_UP
    ENEMY_SPEED_STEP: 50,       // px/s — how much SPEED_UP/SLOW_DOWN changes
    ENEMY_JUMP_FORCE: -450,     // px/s — negative = upward (canvas Y increases downward)
    ENEMY_COLOR: '#e74c3c',

    // === Stones ===
    STONE_RADIUS: 10,           // px
    STONE_COLOR: '#95a5a6',
    STONE_SPAWN_COOLDOWN: 500,  // ms — minimum time between spawns

    // === Physics ===
    GRAVITY: 800,               // px/s² — acceleration due to gravity

    // === AI ===
    AI_DECISION_INTERVAL: 150,  // ms — how often the AI picks a new action

    // === Game Flow ===
    ROUND_RESET_DELAY: 1500,    // ms — pause between rounds
};

/**
 * Action Space — the set of discrete actions the enemy can take.
 * 
 * WHY DISCRETE ACTIONS?
 * In Reinforcement Learning, the agent must choose from a finite set of actions.
 * This is called a "discrete action space". The alternative is a "continuous
 * action space" (e.g., choose exact velocity), which requires different algorithms
 * (like Policy Gradient methods). We start discrete for simplicity.
 * 
 * Each action maps to a specific behavior in Enemy.applyAction().
 */
export const ACTIONS = {
    MOVE_LEFT:  'MOVE_LEFT',    // Move left at current speed
    MOVE_RIGHT: 'MOVE_RIGHT',  // Move right at current speed
    JUMP:       'JUMP',         // Jump (only if on ground)
    SPEED_UP:   'SPEED_UP',    // Increase movement speed
    SLOW_DOWN:  'SLOW_DOWN',   // Decrease movement speed
    STAY:       'STAY',         // Do nothing (stop horizontal movement)
};

// Array form for random selection and indexing
export const ACTION_LIST = Object.values(ACTIONS);
