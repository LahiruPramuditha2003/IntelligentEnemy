/**
 * effects.js — Visual Particle & Screen FX System
 * 
 * Provides dynamic particle explosions, trails, and sparks.
 */

export class Particle {
    constructor(x, y, vx, vy, color, radius, maxLife, decayRate = 1.0) {
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.color = color;
        this.radius = radius;
        this.life = maxLife;
        this.maxLife = maxLife;
        this.decayRate = decayRate;
    }

    update(dt) {
        this.x += this.vx * dt;
        this.y += this.vy * dt;
        // Gravity on particles
        this.vy += 300 * dt;
        // Drag
        this.vx *= 0.96;
        this.life -= dt * this.decayRate;
    }

    draw(ctx) {
        if (this.life <= 0) return;
        const alpha = Math.max(0, this.life / this.maxLife);
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, Math.max(0.5, this.radius * alpha), 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }
}

export class EffectsManager {
    constructor() {
        this.particles = [];
    }

    /**
     * Burst of fiery sparks on enemy defeat.
     */
    createDefeatExplosion(x, y, width, height) {
        const centerX = x + width / 2;
        const centerY = y + height / 2;
        const colors = ['#e74c3c', '#e67e22', '#f1c40f', '#ffffff', '#ff7675'];

        for (let i = 0; i < 40; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = 80 + Math.random() * 260;
            const vx = Math.cos(angle) * speed;
            const vy = Math.sin(angle) * speed - 60; // Bias upward
            const color = colors[Math.floor(Math.random() * colors.length)];
            const radius = 2 + Math.random() * 4;
            const life = 0.5 + Math.random() * 0.6;

            this.particles.push(new Particle(centerX, centerY, vx, vy, color, radius, life, 1.2));
        }
    }

    /**
     * Small dust burst when stone strikes ground.
     */
    createGroundDust(x, y) {
        for (let i = 0; i < 8; i++) {
            const vx = (Math.random() - 0.5) * 80;
            const vy = -(20 + Math.random() * 50);
            const radius = 1.5 + Math.random() * 2;
            const life = 0.3 + Math.random() * 0.3;
            this.particles.push(new Particle(x, y, vx, vy, '#7f8c8d', radius, life, 1.8));
        }
    }

    /**
     * Jump dust beneath enemy feet.
     */
    createJumpDust(x, y, width) {
        for (let i = 0; i < 6; i++) {
            const vx = (Math.random() - 0.5) * 60;
            const vy = -(10 + Math.random() * 25);
            this.particles.push(new Particle(x + width / 2, y, vx, vy, '#4ecdc4', 1.5, 0.25, 2.0));
        }
    }

    update(dt) {
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            p.update(dt);
            if (p.life <= 0) {
                this.particles.splice(i, 1);
            }
        }
    }

    draw(ctx) {
        for (const p of this.particles) {
            p.draw(ctx);
        }
    }

    reset() {
        this.particles = [];
    }
}
