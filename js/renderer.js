/**
 * renderer.js — Canvas rendering engine.
 * 
 * Responsible for drawing everything visible on screen:
 * - Background and ground
 * - Enemy (rectangle with simple face)
 * - Stones (circles with highlight)
 * - Danger indicators (landing prediction lines)
 * - UI overlay (round, stats, messages)
 * 
 * All drawing uses the Canvas 2D API. We keep rendering separate from
 * game logic (Single Responsibility Principle) so we can later:
 * - Add visual effects without touching game code
 * - Switch to WebGL if performance demands it
 * - Run the game "headless" (no rendering) for fast AI training
 */

import { CONFIG } from './config.js';

export class Renderer {
    /**
     * @param {CanvasRenderingContext2D} ctx
     */
    constructor(ctx) {
        this.ctx = ctx;
    }

    /**
     * Clear the canvas with the background color.
     */
    clear() {
        const ctx = this.ctx;
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, CONFIG.CANVAS_WIDTH, CONFIG.CANVAS_HEIGHT);
    }

    /**
     * Draw the ground area and edge line.
     */
    drawGround() {
        const ctx = this.ctx;

        // Ground fill
        ctx.fillStyle = '#16213e';
        ctx.fillRect(0, CONFIG.GROUND_Y, CONFIG.CANVAS_WIDTH, CONFIG.CANVAS_HEIGHT - CONFIG.GROUND_Y);

        // Ground edge line (subtle)
        ctx.strokeStyle = '#0f3460';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, CONFIG.GROUND_Y);
        ctx.lineTo(CONFIG.CANVAS_WIDTH, CONFIG.GROUND_Y);
        ctx.stroke();
    }

    /**
     * Draw the enemy entity.
     * Alive: colored rectangle with simple face
     * Dead: red flash effect
     * 
     * @param {Enemy} enemy
     */
    /**
     * Draw the enemy entity with cybernetic details and stone-tracking visor.
     * @param {Enemy} enemy
     * @param {Stone[]} stones - active stones for tracking gaze
     */
    drawEnemy(enemy, stones = []) {
        const ctx = this.ctx;

        if (!enemy.isAlive) {
            // Death effect: expanding red glow
            ctx.fillStyle = 'rgba(231, 76, 60, 0.4)';
            ctx.beginPath();
            ctx.arc(enemy.x + enemy.width / 2, enemy.y + enemy.height / 2, enemy.width, 0, Math.PI * 2);
            ctx.fill();
            return;
        }

        // Find nearest stone to track gaze
        let nearestStone = null;
        let minDist = Infinity;
        const enemyCenterX = enemy.x + enemy.width / 2;
        const enemyCenterY = enemy.y + enemy.height / 2;

        for (const s of stones) {
            if (!s.active) continue;
            const dist = Math.hypot(s.x - enemyCenterX, s.y - enemyCenterY);
            if (dist < minDist) {
                minDist = dist;
                nearestStone = s;
            }
        }

        const isDangerClose = minDist < 160;

        // --- Thruster Flame if Jumping ---
        if (enemy.isJumping) {
            ctx.fillStyle = '#3498db';
            ctx.beginPath();
            ctx.moveTo(enemy.x + 8, enemy.y + enemy.height);
            ctx.lineTo(enemy.x + enemy.width / 2, enemy.y + enemy.height + 12 + Math.random() * 6);
            ctx.lineTo(enemy.x + enemy.width - 8, enemy.y + enemy.height);
            ctx.fill();

            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.moveTo(enemy.x + 12, enemy.y + enemy.height);
            ctx.lineTo(enemy.x + enemy.width / 2, enemy.y + enemy.height + 6);
            ctx.lineTo(enemy.x + enemy.width - 12, enemy.y + enemy.height);
            ctx.fill();
        }

        // --- Droid Chassis (Rounded Rect) ---
        ctx.save();
        ctx.fillStyle = isDangerClose ? '#962d22' : '#2c3e50';
        ctx.strokeStyle = isDangerClose ? '#e74c3c' : '#4ecdc4';
        ctx.lineWidth = 2;

        // Rounded chassis
        const radius = 6;
        ctx.beginPath();
        ctx.roundRect(enemy.x, enemy.y, enemy.width, enemy.height, radius);
        ctx.fill();
        ctx.stroke();

        // Inner armor plate
        ctx.fillStyle = '#1e272e';
        ctx.beginPath();
        ctx.roundRect(enemy.x + 3, enemy.y + 3, enemy.width - 6, enemy.height - 6, 4);
        ctx.fill();

        // --- Cyber Visor / Eye ---
        const visorY = enemy.y + 10;
        const visorWidth = enemy.width - 10;
        const visorHeight = 8;
        ctx.fillStyle = '#0f171e';
        ctx.beginPath();
        ctx.roundRect(enemy.x + 5, visorY, visorWidth, visorHeight, 3);
        ctx.fill();

        // Tracking pupil / scanner dot inside visor
        let pupilOffsetRatio = 0;
        if (nearestStone) {
            const dx = nearestStone.x - enemyCenterX;
            pupilOffsetRatio = Math.max(-1, Math.min(1, dx / 200));
        }
        const maxPupilShift = (visorWidth / 2) - 4;
        const pupilX = (enemy.x + 5 + visorWidth / 2) + pupilOffsetRatio * maxPupilShift;

        // Visor glow color: Red when danger, cyan when safe
        const glowColor = isDangerClose ? '#ff3838' : '#00d8d6';
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = isDangerClose ? 10 : 6;

        ctx.fillStyle = glowColor;
        ctx.beginPath();
        ctx.arc(pupilX, visorY + visorHeight / 2, 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();

        // --- Speed Indicator Bar ---
        const speedRatio = (enemy.speed - CONFIG.ENEMY_MIN_SPEED) /
                           (CONFIG.ENEMY_MAX_SPEED - CONFIG.ENEMY_MIN_SPEED);
        const barWidth = enemy.width * speedRatio;
        ctx.fillStyle = `hsl(${120 * (1 - speedRatio)}, 90%, 50%)`;
        ctx.fillRect(enemy.x, CONFIG.GROUND_Y + 4, barWidth, 3);
    }

    /**
     * Draw stones with dynamic glowing halo and trail.
     * @param {Stone[]} stones
     */
    drawStones(stones) {
        const ctx = this.ctx;

        for (const stone of stones) {
            if (!stone.active) continue;

            // Falling tail glow
            const tailLength = Math.min(45, stone.vy * 0.06);
            if (tailLength > 4) {
                const grad = ctx.createLinearGradient(stone.x, stone.y - tailLength, stone.x, stone.y);
                grad.addColorStop(0, 'rgba(243, 156, 18, 0)');
                grad.addColorStop(1, 'rgba(243, 156, 18, 0.45)');
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.moveTo(stone.x - stone.radius * 0.8, stone.y);
                ctx.lineTo(stone.x, stone.y - tailLength);
                ctx.lineTo(stone.x + stone.radius * 0.8, stone.y);
                ctx.closePath();
                ctx.fill();
            }

            // Stone body
            ctx.save();
            ctx.shadowColor = '#f39c12';
            ctx.shadowBlur = 8;
            ctx.fillStyle = '#ecf0f1';
            ctx.beginPath();
            ctx.arc(stone.x, stone.y, stone.radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();

            // Inner core
            ctx.fillStyle = '#bdc3c7';
            ctx.beginPath();
            ctx.arc(stone.x - 2, stone.y - 2, stone.radius * 0.45, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    /**
     * Draw landing prediction indicators.
     * Dashed vertical lines show where each stone will land.
     * This helps the player aim and makes the game more readable.
     * 
     * @param {Stone[]} stones
     */
    drawDangerIndicators(stones) {
        const ctx = this.ctx;

        for (const stone of stones) {
            if (!stone.active) continue;

            // Dashed line from stone to ground
            ctx.strokeStyle = 'rgba(231, 76, 60, 0.12)';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 6]);
            ctx.beginPath();
            ctx.moveTo(stone.x, stone.y + stone.radius);
            ctx.lineTo(stone.x, CONFIG.GROUND_Y);
            ctx.stroke();
            ctx.setLineDash([]);

            // Landing point dot
            ctx.fillStyle = 'rgba(231, 76, 60, 0.25)';
            ctx.beginPath();
            ctx.arc(stone.x, CONFIG.GROUND_Y, 4, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    /**
     * Draw the UI overlay (stats, messages, hints).
     * @param {Game} game
     */
    drawUI(game) {
        const ctx = this.ctx;

        // --- Top-left: Round & survival time ---
        ctx.fillStyle = '#b0b0b0';
        ctx.font = '14px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`Round: ${game.round}`, 15, 22);
        ctx.fillText(`Survival: ${game.stats.currentSurvivalTime.toFixed(1)}s`, 15, 40);

        // --- Top-right: Win count & best time ---
        ctx.textAlign = 'right';
        ctx.fillText(`Wins: ${game.stats.playerWins}`, CONFIG.CANVAS_WIDTH - 15, 22);
        ctx.fillText(`Best: ${game.stats.longestSurvival.toFixed(1)}s`, CONFIG.CANVAS_WIDTH - 15, 40);

        // --- Top-center: AI brain name & learning stats ---
        ctx.textAlign = 'center';
        if (game.aiBrain) {
            ctx.fillStyle = '#4ecdc4';
            ctx.font = 'bold 14px monospace';
            ctx.fillText(`AI: ${game.aiBrain.name}`, CONFIG.CANVAS_WIDTH / 2, 22);

            if (typeof game.aiBrain.getStats === 'function') {
                const aiStats = game.aiBrain.getStats();
                ctx.fillStyle = '#f39c12';
                ctx.font = '12px monospace';
                ctx.fillText(`ε: ${aiStats.epsilon} | States: ${aiStats.statesDiscovered} | Episodes: ${aiStats.roundsTrained}`, CONFIG.CANVAS_WIDTH / 2, 38);
            }
        }

        // Average survival (rolling)
        const avgSurvival = game.getAverageSurvival();
        if (game.stats.totalRounds > 0) {
            ctx.fillStyle = '#888';
            ctx.font = '12px monospace';
            const yPos = (game.aiBrain && typeof game.aiBrain.getStats === 'function') ? 54 : 40;
            ctx.fillText(`Avg: ${avgSurvival.toFixed(1)}s`, CONFIG.CANVAS_WIDTH / 2, yPos);
        }

        // --- Round Over message ---
        if (game.state === 'ROUND_OVER') {
            // Semi-transparent overlay
            ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            ctx.fillRect(0, 0, CONFIG.CANVAS_WIDTH, CONFIG.CANVAS_HEIGHT);

            ctx.fillStyle = '#e74c3c';
            ctx.font = 'bold 36px monospace';
            ctx.textAlign = 'center';
            ctx.fillText('DEFEATED!', CONFIG.CANVAS_WIDTH / 2, CONFIG.CANVAS_HEIGHT / 2 - 20);

            ctx.fillStyle = '#999';
            ctx.font = '14px monospace';
            ctx.fillText(
                `Survived ${game.stats.currentSurvivalTime.toFixed(1)}s — Next round starting...`,
                CONFIG.CANVAS_WIDTH / 2,
                CONFIG.CANVAS_HEIGHT / 2 + 15
            );
        }

        // --- First-round hint ---
        if (game.state === 'PLAYING' && game.stoneManager.stones.length === 0 && game.stats.totalRounds === 0) {
            ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
            ctx.font = '16px monospace';
            ctx.textAlign = 'center';
            ctx.fillText('☝ Click above to drop stones!', CONFIG.CANVAS_WIDTH / 2, CONFIG.CANVAS_HEIGHT / 2 - 100);
        }
    }

    /**
     * Master render — called once per frame.
     * Draw order matters (back to front):
     * 1. Background (clear)
     * 2. Ground
     * 3. Danger indicators (behind stones)
     * 4. Stones
     * 5. Particles & FX
     * 6. Enemy (with tracking visor)
     * 7. UI overlay (on top of everything)
     * 
     * @param {Game} game
     * @param {EffectsManager} [effectsManager]
     */
    render(game, effectsManager = null) {
        this.clear();
        this.drawGround();
        const activeStones = game.stoneManager.getActiveStones();
        this.drawDangerIndicators(activeStones);
        this.drawStones(activeStones);
        if (effectsManager) {
            effectsManager.draw(this.ctx);
        }
        this.drawEnemy(game.enemy, activeStones);
        this.drawUI(game);
    }
}
