/**
 * physics.js — Collision detection utilities.
 * 
 * COLLISION DETECTION APPROACH: Circle vs Rectangle
 * 
 * Why this approach?
 * - Stones are circles, the enemy is a rectangle
 * - AABB (Axis-Aligned Bounding Box) would treat the stone as a rectangle too,
 *   causing false positives at corners
 * - Circle-Rect is precise and still fast (no square root needed — we compare
 *   squared distances)
 * 
 * Algorithm:
 * 1. Find the point on the rectangle closest to the circle's center
 * 2. If the distance from that point to the circle's center <= radius, collision!
 */

/**
 * Check if a circle and axis-aligned rectangle overlap.
 * @param {{x: number, y: number, radius: number}} circle 
 * @param {{x: number, y: number, width: number, height: number}} rect 
 * @returns {boolean}
 */
export function checkCircleRectCollision(circle, rect) {
    // Clamp the circle center to the rectangle bounds to find the closest point
    const closestX = Math.max(rect.x, Math.min(circle.x, rect.x + rect.width));
    const closestY = Math.max(rect.y, Math.min(circle.y, rect.y + rect.height));

    // Compute squared distance (avoiding expensive sqrt)
    const dx = circle.x - closestX;
    const dy = circle.y - closestY;
    const distanceSquared = dx * dx + dy * dy;

    // Collision if distance <= radius (compare squared to avoid sqrt)
    return distanceSquared <= circle.radius * circle.radius;
}

/**
 * Check if a stone has hit the enemy.
 * @param {Stone} stone 
 * @param {Enemy} enemy 
 * @returns {boolean}
 */
export function checkStoneEnemyCollision(stone, enemy) {
    if (!stone.active || !enemy.isAlive) return false;

    const circle = { x: stone.x, y: stone.y, radius: stone.radius };
    const rect = enemy.getBounds();

    return checkCircleRectCollision(circle, rect);
}
