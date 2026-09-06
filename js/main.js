/**
 * main.js — Game entry point and orchestrator.
 * 
 * This file wires everything together:
 * 1. Initializes the canvas and 2D context
 * 2. Creates game entities and systems
 * 3. Sets up the AI brain (defaulting to Random)
 * 4. Runs the game loop
 * 
 * THE GAME LOOP:
 * Every frame (~60fps), the loop does:
 *   1. Calculate delta time (time since last frame)
 *   2. Process player input (clicks → stones)
 *   3. Update game state (AI decision, physics, collision)
 *   4. Render everything to the canvas
 *   5. Request the next frame
 * 
 * This is the standard game loop pattern used in virtually all games,
 * from simple browser games to AAA titles. The key insight is that
 * the game is a simulation that advances in discrete time steps.
 */

import { CONFIG } from './config.js';
import { Game } from './game.js';
import { Renderer } from './renderer.js';
import { InputHandler } from './input.js';
import { RandomBrain } from './ai/random.js';
import { HeuristicBrain } from './ai/heuristic.js';
import { QLearningBrain } from './ai/qlearning.js';
import { DQNBrain } from './ai/dqn.js';
import { Enemy } from './enemy.js';
import { StoneManager } from './stone.js';
import { checkStoneEnemyCollision } from './physics.js';
import { EffectsManager } from './effects.js';
import { SoundManager } from './audio.js';

// === Initialization ===

const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
canvas.width = CONFIG.CANVAS_WIDTH;
canvas.height = CONFIG.CANVAS_HEIGHT;

// Create core systems
const game = new Game();
const renderer = new Renderer(ctx);
const input = new InputHandler(canvas);
const effects = new EffectsManager();
const audio = new SoundManager();

// Game Speed Multiplier
let speedMultiplier = 1.0;

// Wire visual and audio event callbacks
game.onHitCallback = (x, y) => {
    effects.createDefeatExplosion(x, y, game.enemy.width, game.enemy.height);
    audio.playDefeat();
};

game.onGroundHitCallback = (x, y) => {
    effects.createGroundDust(x, y);
    audio.playGroundThud();
};

// === AI Brain Registry ===
// All available brains are registered here.
const brains = {
    random: new RandomBrain(),
    heuristic: new HeuristicBrain(),
    qlearning: new QLearningBrain(),
    dqn: new DQNBrain(),
};

// Set the default brain
game.setAIBrain(brains.random);

// === Dashboard UI Elements ===
const statEpsilon = document.getElementById('statEpsilon');
const statStates = document.getElementById('statStates');
const statEpisodes = document.getElementById('statEpisodes');
const statUpdates = document.getElementById('statUpdates');

function updateDashboardUI() {
    if (game.aiBrain && typeof game.aiBrain.getStats === 'function') {
        const stats = game.aiBrain.getStats();
        if (statEpsilon) statEpsilon.textContent = stats.epsilon;
        if (statStates) statStates.textContent = stats.statesDiscovered;
        if (statEpisodes) statEpisodes.textContent = stats.roundsTrained;
        if (statUpdates) statUpdates.textContent = stats.totalUpdates;
    } else {
        if (statEpsilon) statEpsilon.textContent = 'N/A';
        if (statStates) statStates.textContent = 'N/A';
        if (statEpisodes) statEpisodes.textContent = 'N/A';
        if (statUpdates) statUpdates.textContent = 'N/A';
    }
}

// === UI: Brain Selector ===
const brainSelector = document.getElementById('brainSelector');
if (brainSelector) {
    brainSelector.addEventListener('change', (e) => {
        const selectedBrain = brains[e.target.value];
        if (selectedBrain) {
            game.setAIBrain(selectedBrain);
            // Reset the game when switching brains for a fair comparison
            game.round = 1;
            game.state = 'PLAYING';
            game.enemy.reset();
            game.stoneManager.reset();
            game.stats = {
                totalRounds: 0,
                playerWins: 0,
                currentSurvivalTime: 0,
                longestSurvival: 0,
                recentSurvivals: [],
            };
            updateDashboardUI();
        }
    });
}

// === Headless Fast-Training Function ===
function runFastTraining(numRounds = 25) {
    const brain = game.aiBrain;
    const isRLBrain = brain && typeof brain.onDeath === 'function' && (brain.qTable || brain.policyNet);
    if (!isRLBrain) {
        alert('Fast training is available for Reinforcement Learning brains (Q-Learning or DQN). Please select one first!');
        return;
    }

    const dt = 1 / 60;
    const maxTime = 20; // max seconds per simulated round

    for (let r = 0; r < numRounds; r++) {
        const simEnemy = new Enemy();
        const simStones = new StoneManager();
        let time = 0;
        let lastDecision = 0;
        let nextSpawnTime = 0.4;

        brain.reset();

        while (time < maxTime && simEnemy.isAlive) {
            time += dt;

            // Spawn stones near enemy with random variance
            if (time >= nextSpawnTime) {
                const targetX = Math.random() < 0.7
                    ? (simEnemy.x + simEnemy.width / 2 + (Math.random() * 60 - 30))
                    : Math.random() * CONFIG.CANVAS_WIDTH;
                simStones.spawn(Math.max(20, Math.min(CONFIG.CANVAS_WIDTH - 20, targetX)), time * 1000);
                nextSpawnTime += 0.6 + Math.random() * 0.5;
            }

            // AI decision
            if (time * 1000 - lastDecision >= CONFIG.AI_DECISION_INTERVAL) {
                const simState = {
                    enemyX: simEnemy.x,
                    enemyY: simEnemy.y,
                    enemyVX: simEnemy.vx,
                    enemyVY: simEnemy.vy,
                    enemySpeed: simEnemy.speed,
                    enemyIsJumping: simEnemy.isJumping,
                    stones: simStones.getActiveStones(),
                    canvasWidth: CONFIG.CANVAS_WIDTH,
                    groundY: CONFIG.GROUND_Y,
                };
                const action = brain.decide(simState);
                simEnemy.applyAction(action);
                lastDecision = time * 1000;
            }

            // Physics update
            simEnemy.update(dt);
            simStones.update(dt);

            // Collision check
            for (const stone of simStones.getActiveStones()) {
                if (checkStoneEnemyCollision(stone, simEnemy)) {
                    simEnemy.die();
                    brain.onDeath({
                        enemyX: simEnemy.x,
                        enemyY: simEnemy.y,
                        enemyVX: simEnemy.vx,
                        enemyVY: simEnemy.vy,
                        enemySpeed: simEnemy.speed,
                        enemyIsJumping: simEnemy.isJumping,
                        stones: simStones.getActiveStones(),
                        canvasWidth: CONFIG.CANVAS_WIDTH,
                        groundY: CONFIG.GROUND_Y,
                    });
                    break;
                }
            }
        }
    }

    updateDashboardUI();
}

// === Button Event Listeners ===
const btnFastTrain = document.getElementById('btnFastTrain');
if (btnFastTrain) {
    btnFastTrain.addEventListener('click', () => {
        if (brainSelector.value !== 'qlearning' && brainSelector.value !== 'dqn') {
            brainSelector.value = 'dqn';
            brainSelector.dispatchEvent(new Event('change'));
        }
        runFastTraining(25);
    });
}

const btnResetAI = document.getElementById('btnResetAI');
if (btnResetAI) {
    btnResetAI.addEventListener('click', () => {
        if (game.aiBrain && typeof game.aiBrain.clearKnowledge === 'function') {
            if (confirm('Are you sure you want to erase all learned Q-values and reset exploration?')) {
                game.aiBrain.clearKnowledge();
                updateDashboardUI();
            }
        } else {
            alert('Active brain has no memory to reset.');
        }
    });
}

// === Speed Selector ===
const speedSelector = document.getElementById('speedSelector');
if (speedSelector) {
    speedSelector.addEventListener('change', (e) => {
        speedMultiplier = parseFloat(e.target.value) || 1.0;
    });
}

// === Audio Toggle Button ===
const btnSoundToggle = document.getElementById('btnSoundToggle');
if (btnSoundToggle) {
    btnSoundToggle.textContent = audio.isMuted ? '🔇 Sound: OFF' : '🔊 Sound: ON';
    btnSoundToggle.addEventListener('click', () => {
        const muted = audio.toggleMute();
        btnSoundToggle.textContent = muted ? '🔇 Sound: OFF' : '🔊 Sound: ON';
    });
}

// === Game Loop ===

let lastTime = 0;
let frameCount = 0;

/**
 * The main game loop.
 * 
 * Uses requestAnimationFrame for smooth 60fps rendering.
 * Delta time is capped at 50ms to prevent physics explosions
 * when the tab is backgrounded (browser throttles rAF).
 * 
 * @param {number} timestamp - High-resolution timestamp from rAF (ms)
 */
function gameLoop(timestamp) {
    // Delta time in seconds, capped to prevent spiral of death
    const rawDt = Math.min((timestamp - lastTime) / 1000, 0.05);
    lastTime = timestamp;

    const simDt = rawDt * speedMultiplier;

    // 1. Process player input
    const clicks = input.getClicks();
    for (const click of clicks) {
        game.spawnStone(click.x, timestamp);
        audio.playSpawn();
    }

    // 2. Update physics, AI & particles
    game.update(simDt, timestamp);
    effects.update(simDt);

    // 3. Render with effects
    renderer.render(game, effects);

    // Update dashboard every 20 frames (~3 times per second)
    frameCount++;
    if (frameCount % 20 === 0) {
        updateDashboardUI();
    }

    // 4. Request next frame
    requestAnimationFrame(gameLoop);
}

// Initial dashboard sync
updateDashboardUI();

// === Start the game ===
// We use a wrapper to properly initialize lastTime on the first frame
requestAnimationFrame((timestamp) => {
    lastTime = timestamp;
    gameLoop(timestamp);
});
