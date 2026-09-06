# 🪨 Stone Drop: AI Enemy Learning Game

> A browser-based AI research and demonstration game where an enemy droid learns to dodge falling stones through trial, error, and reinforcement learning — evolving from a naive random agent into an intelligent, evasive adversary.

[![Tech Stack: Vanilla JavaScript](https://img.shields.io/badge/Language-Vanilla%20ES6%20JS-F7DF1E?logo=javascript&logoColor=black)](#technologies-used)
[![Zero External Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(No%20Libraries)-brightgreen)](#zero-dependency-philosophy)
[![Canvas: HTML5 2D](https://img.shields.io/badge/Graphics-HTML5%20Canvas%202D-E34F26?logo=html5&logoColor=white)](#graphics-and-rendering)
[![Audio: Web Audio API](https://img.shields.io/badge/Audio-Procedural%20Web%20Audio-blue)](#audio-synthesizer)
[![AI: Q-Learning & DQN from Scratch](https://img.shields.io/badge/AI-Tabular%20RL%20%2B%20DQN%20from%20Scratch-8A2BE2)](#ai-models-implemented)

---

## 📌 Project Overview

**Stone Drop** is an interactive web game and artificial intelligence laboratory. The player clicks in the sky to spawn stones that accelerate downward under simulated gravitational physics. An AI-controlled enemy droid moves along the ground, jumping and adjusting its speed to avoid impact.

The central research focus is **comparative AI progression**:
- How does a hand-crafted kinematic heuristic compare to self-taught Reinforcement Learning?
- How does tabular Q-learning perform under state discretization compared to Deep Q-Networks (DQN) with continuous function approximation?
- How can complex AI algorithms (matrix operations, analytical backpropagation, Adam optimization, experience replay, target networks) be implemented entirely from first principles without external libraries like TensorFlow or PyTorch?

---

## 🎮 Core Gameplay & Interactive Features

- **Stone Dropping Mechanics**: Click anywhere in the upper canvas to drop stones with customizable landing corridors and gravity acceleration ($800\text{ px/s}^2$).
- **Swappable AI Brains**: Hot-swap between **Random**, **Heuristic**, **Q-Learning**, and **DQN** on the fly using the **Strategy Pattern**.
- **Interactive Cyber Droid**:
  - Gaze-tracking cyber visor: The droid's glowing pupil physically tracks the $(x, y)$ coordinates of falling stones in real time.
  - Threat detection: The visor pulses crimson red when a stone breaches critical proximity ($< 160\text{ px}$).
  - Thruster flames: Dual blue-white propulsion flares ignite whenever the droid jumps.
- **Procedural Audio Synthesizer**: Generates dynamic whooshes, landing thuds, and defeat chords using the native **Web Audio API** with a mute toggle.
- **Particle Explosion System**: 40-spark radial bursts on defeat, dust puffs on ground impact, and propulsion sparks.
- **Live Speed Controls**: Play at **1x (Normal)**, **2x (Fast)**, or **5x (Turbo)** speed to observe AI training in real-time acceleration.
- **⚡ Fast-Train Mode**: Run 25 rapid headless simulation rounds in milliseconds to train the RL agents on demand.
- **Live AI Dashboard**: Real-time telemetry monitoring exploration rate ($\epsilon$), discovered states, replay buffer size, training episodes, and gradient updates.
- **Memory Persistence**: Automatically saves and restores learned Q-tables and neural network weights via `localStorage`.

---

## 🧠 AI Models Implemented

| Model | Category | State Representation | Optimization / Decision | Key Characteristics |
|---|---|---|---|---|
| **Random** | Baseline | None | Uniform Random ($P(a) = \frac{1}{6}$) | Lower-bound baseline; erratic, jittery movement |
| **Heuristic** | Expert System | Continuous $(x, y, v_y)$ | Analytical Kinematic TTI Solver | Solves $\frac{1}{2}gt^2 + v_y t + \Delta y = 0$; dodges threats; center-seeking bias |
| **Q-Learning** | Tabular RL | 144 Binned States ($6 \times 4 \times 3 \times 2$) | 1-Step Temporal Difference TD(0) | Discretized state space; $\epsilon$-greedy policy; table of 864 Q-values |
| **DQN** | Deep RL | 7 Continuous Normalized Features | Deep MLP + Adam Optimizer from Scratch | 2-hidden-layer ANN; Experience Replay Buffer; Target Network; continuous control |

---

## 📊 Empirical Performance Benchmark

Automated head-to-head evaluation over 50–100 simulated rounds per model:

```
Average Survival Time (Seconds per Round)
────────────────────────────────────────────────────────────────────────
Random (Baseline)       █ 7.04s
Heuristic (Rule-Based)  ████████████████ 16.73s
Q-Learning (Untrained)  █ 6.78s
Q-Learning (Trained)    ███████████████ 15.12s   (+123% improvement)
DQN (Untrained)         █ 5.10s
DQN (Trained)           █████████████████ 17.13s (+236% improvement)
────────────────────────────────────────────────────────────────────────
```

Both Reinforcement Learning models exhibit clear self-improvement:
- **Tabular Q-Learning** learned to avoid `OVERHEAD | LOW` states, elevating survival from **6.78s** to **15.12s** (**2.2x**).
- **Deep Q-Network** generalized non-linear escape vectors across continuous space, elevating survival from **5.10s** to **17.13s** (**3.3x**), successfully matching and surpassing the human-engineered heuristic!

---

## 🏗️ Architecture & File Structure

The project strictly adheres to modular software engineering principles:

```
Attempt2/
├── index.html              # Entry point, HUD controls, canvas viewport, AI dashboard
├── css/
│   └── style.css           # Dark cybernetic theme, glassmorphic HUD cards, layout
├── js/
│   ├── main.js             # Game orchestrator, animation loop, fast-training runner
│   ├── config.js           # Central game constants, physics values, discrete action space
│   ├── game.js             # State coordinator, round lifecycle, state extraction
│   ├── renderer.js         # Canvas 2D engine: cyber-droid, glowing stones, trajectories
│   ├── physics.js          # Precision circle-rectangle collision detection
│   ├── enemy.js            # Enemy entity physics, Euler integration, boundary clamping
│   ├── stone.js            # Stone entity kinematics and lifecycle management
│   ├── input.js            # Decoupled mouse input queue
│   ├── effects.js          # Particle physics engine (explosions, dust, thrust flares)
│   ├── audio.js            # Procedural Web Audio API sound synthesizer
│   └── ai/
│       ├── brain.js        # Base AIBrain interface (Strategy Pattern contract)
│       ├── random.js       # Baseline random agent
│       ├── heuristic.js    # Rule-based expert system (kinematic TTI solver)
│       ├── qlearning.js    # Tabular Q-Learning with state discretization
│       ├── neural_net.js   # Multi-layer Perceptron (forward/backprop/Adam) from scratch
│       └── dqn.js          # Deep Q-Network with Experience Replay & Target Network
├── README.md               # Quickstart, overview, and usage documentation
├── PROJECT_EXPLANATION.md  # Exhaustive mathematical, theoretical, and architectural guide
└── .gitignore              # Ignored files for version control
```

---

## 🚀 Quickstart & Setup

Because Stone Drop uses standard **ES6 Modules**, it must be served over HTTP rather than opened as a local `file://` URL.

### Option 1: Using Node.js / npx (Recommended)
```bash
# Clone or navigate to the project directory
cd Attempt2

# Serve instantly using npx http-server
npx -y http-server . -p 8080 -c-1
```
Open your browser and navigate to **`http://localhost:8080`**.

### Option 2: Using Python 3
```bash
cd Attempt2
python -m http.server 8080
```
Open **`http://localhost:8080`**.

### Option 3: Using VS Code Live Server
1. Open the `Attempt2` folder in Visual Studio Code.
2. Install the **Live Server** extension by Ritwick Dey.
3. Click **"Go Live"** in the bottom status bar.

---

## 🕹️ How to Play & Experiment

1. **Select an AI Model**: Use the **"AI Brain"** dropdown to switch between `Random`, `Heuristic`, `Q-Learning`, and `DQN`.
2. **Drop Stones**: Click anywhere above the ground to drop a stone. Trajectory preview lines assist targeting.
3. **Control Speed**: Switch between `1x`, `2x`, and `5x` to observe live learning at accelerated rates.
4. **Fast-Forward Training**: Select `Q-Learning` or `DQN` and click **"⚡ Fast Train (25 Rounds)"** to simulate 25 headless battles in seconds. Watch the exploration rate ($\epsilon$) decay and survival time rise!
5. **Reset Knowledge**: Click **"↺ Reset AI Memory"** to clear the neural weights or Q-table and re-observe training from scratch.
6. **Sound Effects**: Click **"🔊 Sound: ON/OFF"** to toggle the procedural audio synthesizer.

---

## 📖 In-Depth Learning Guide

For an exhaustive, step-by-step mathematical breakdown of the kinematics, backpropagation formulas, Bellman optimality proofs, and state discretization theory, read:

👉 **[PROJECT_EXPLANATION.md](file:///c:/Users/lahir/Desktop/Attempt2/PROJECT_EXPLANATION.md)**

---

## 📜 License & Academic Usage

This project was developed as an educational and research exploration of Reinforcement Learning and Artificial Neural Networks from first principles. Feel free to use, adapt, and extend this codebase for academic research, coursework, or demonstrations.
