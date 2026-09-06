/**
 * audio.js — Procedural Sound Synthesizer via Web Audio API
 * 
 * Generates all sound effects mathematically (sine/triangle oscillators,
 * white noise, and frequency ramps) without requiring external audio files.
 */

export class SoundManager {
    constructor() {
        this.ctx = null;
        this.isMuted = false;
        try {
            this.isMuted = localStorage.getItem('stone_drop_muted') === 'true';
        } catch (e) {
            // Storage restricted
        }
    }

    /**
     * Lazy init AudioContext on first user interaction (browser requirement).
     */
    init() {
        if (!this.ctx) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                this.ctx = new AudioCtx();
            }
        }
        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }

    toggleMute() {
        this.isMuted = !this.isMuted;
        try {
            localStorage.setItem('stone_drop_muted', this.isMuted);
        } catch (e) {
            // Ignore
        }
        return this.isMuted;
    }

    /**
     * Whoosh sound when dropping stone.
     */
    playSpawn() {
        if (this.isMuted) return;
        this.init();
        if (!this.ctx) return;

        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        osc.type = 'sine';
        osc.frequency.setValueAtTime(600, this.ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(180, this.ctx.currentTime + 0.12);

        gain.gain.setValueAtTime(0.08, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.12);

        osc.connect(gain);
        gain.connect(this.ctx.destination);

        osc.start();
        osc.stop(this.ctx.currentTime + 0.12);
    }

    /**
     * Low thud when stone lands on ground.
     */
    playGroundThud() {
        if (this.isMuted) return;
        this.init();
        if (!this.ctx) return;

        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        osc.type = 'triangle';
        osc.frequency.setValueAtTime(120, this.ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(40, this.ctx.currentTime + 0.1);

        gain.gain.setValueAtTime(0.12, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.1);

        osc.connect(gain);
        gain.connect(this.ctx.destination);

        osc.start();
        osc.stop(this.ctx.currentTime + 0.1);
    }

    /**
     * Explosion sound on enemy defeat.
     */
    playDefeat() {
        if (this.isMuted) return;
        this.init();
        if (!this.ctx) return;

        // Low boom + frequency drop
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(240, this.ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(30, this.ctx.currentTime + 0.35);

        gain.gain.setValueAtTime(0.2, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.35);

        osc.connect(gain);
        gain.connect(this.ctx.destination);

        osc.start();
        osc.stop(this.ctx.currentTime + 0.35);
    }
}
