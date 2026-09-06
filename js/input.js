/**
 * input.js — Player input handling.
 * 
 * Captures mouse clicks on the canvas and queues them for processing
 * by the game loop. We use a queue (pending clicks) rather than
 * processing immediately because:
 * 
 * 1. Input events fire asynchronously (browser event loop)
 * 2. Game logic runs in the game loop (requestAnimationFrame)
 * 3. Processing input during the game loop ensures consistent timing
 * 
 * This separation of "input capture" from "input processing" is a
 * standard game development pattern.
 */

import { CONFIG } from './config.js';

export class InputHandler {
    /**
     * @param {HTMLCanvasElement} canvas - The game canvas element
     */
    constructor(canvas) {
        this.canvas = canvas;
        this.pendingClicks = [];

        // Bind the click listener
        this.canvas.addEventListener('click', (e) => this.handleClick(e));
    }

    /**
     * Handle a mouse click on the canvas.
     * Converts screen coordinates to canvas coordinates and queues the click.
     * 
     * Only clicks above the ground area are registered — you can't drop
     * stones from below the ground!
     * 
     * @param {MouseEvent} e
     */
    handleClick(e) {
        const rect = this.canvas.getBoundingClientRect();

        // Convert screen coordinates to canvas coordinates
        // This accounts for CSS scaling (if canvas display size ≠ internal size)
        const scaleX = CONFIG.CANVAS_WIDTH / rect.width;
        const scaleY = CONFIG.CANVAS_HEIGHT / rect.height;
        const canvasX = (e.clientX - rect.left) * scaleX;
        const canvasY = (e.clientY - rect.top) * scaleY;

        // Only allow clicks above the ground zone
        // The stone will always spawn at y=0 (top) — click just sets X position
        if (canvasY < CONFIG.GROUND_Y - 50) {
            this.pendingClicks.push({ x: canvasX });
        }
    }

    /**
     * Retrieve and clear all pending clicks.
     * Called once per game loop iteration.
     * 
     * @returns {{x: number}[]} Array of click positions
     */
    getClicks() {
        const clicks = [...this.pendingClicks];
        this.pendingClicks = [];
        return clicks;
    }
}
