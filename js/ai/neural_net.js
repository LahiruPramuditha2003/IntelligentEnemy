/**
 * neural_net.js — Multi-Layer Perceptron (Artificial Neural Network) from Scratch
 * 
 * ============================================================================
 * THEORETICAL FOUNDATIONS (Deep Learning / Function Approximation)
 * ============================================================================
 * 
 * 1. Why Neural Networks in Reinforcement Learning?
 *    In Tabular Q-Learning (Phase 3), the Q-table maps discrete states to values.
 *    If state coordinates are continuous, binning causes information loss and
 *    suffers from the Curse of Dimensionality as feature count grows.
 * 
 *    An Artificial Neural Network (ANN) acts as a UNIVERSAL FUNCTION APPROXIMATOR:
 *      Q(s, a; θ) ≈ Q*(s, a)
 *    where θ represents the trainable weights and biases.
 *    The network can take CONTINUOUS normalized coordinates directly and GENERALIZE
 *    to never-before-seen game states!
 * 
 * 2. Mathematical Components Implemented from First Principles:
 *    - Dense (Fully-Connected) Layer:
 *        z = W · x + b
 *        a = activation(z)
 *    - He (Kaiming) Normal Initialization:
 *        Weights drawn from N(0, sqrt(2 / fan_in)) for stable gradients with ReLU.
 *    - Activation Functions:
 *        ReLU(z) = max(0, z)
 *        Linear(z) = z (essential for Q-value outputs which can be any real number)
 *    - Backpropagation & Gradients:
 *        dL/dz = dL/da ⊙ activation'(z)
 *        dL/dW = (dL/dz) · x^T
 *        dL/db = dL/dz
 *        dL/dx = W^T · (dL/dz) (propagated to previous layer)
 *    - Adam Optimizer (Kingma & Ba, 2014):
 *        Adaptive Moment Estimation using first (mean) and second (uncentered
 *        variance) moment vectors with bias correction. Far faster and more stable
 *        than vanilla Stochastic Gradient Descent (SGD).
 */

export class DenseLayer {
    /**
     * @param {number} inputDim - Number of incoming features
     * @param {number} outputDim - Number of neurons in this layer
     * @param {'relu' | 'linear'} activation - Activation function
     */
    constructor(inputDim, outputDim, activation = 'relu') {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.activation = activation;

        // Weights matrix: shape [outputDim][inputDim]
        this.weights = new Float32Array(outputDim * inputDim);
        this.biases = new Float32Array(outputDim);

        // Gradients accumulation
        this.gradW = new Float32Array(outputDim * inputDim);
        this.gradB = new Float32Array(outputDim);

        // Adam Optimizer state
        this.mW = new Float32Array(outputDim * inputDim);
        this.vW = new Float32Array(outputDim * inputDim);
        this.mB = new Float32Array(outputDim);
        this.vB = new Float32Array(outputDim);
        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.epsilon = 1e-8;
        this.t = 0; // Timestep for bias correction

        // Cache for backprop
        this.lastInput = null;
        this.lastZ = new Float32Array(outputDim);
        this.lastA = new Float32Array(outputDim);

        this.initWeights();
    }

    /**
     * He (Kaiming) Normal initialization for ReLU, Glorot Uniform for Linear.
     */
    initWeights() {
        const isRelu = this.activation === 'relu';
        const std = isRelu 
            ? Math.sqrt(2.0 / this.inputDim) 
            : Math.sqrt(2.0 / (this.inputDim + this.outputDim));

        for (let i = 0; i < this.weights.length; i++) {
            // Box-Muller transform for Gaussian random numbers
            const u1 = Math.random() || 1e-7;
            const u2 = Math.random() || 1e-7;
            const randStdNormal = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            this.weights[i] = randStdNormal * std;
        }

        // Small positive bias for ReLU prevents dead neurons at init
        for (let j = 0; j < this.biases.length; j++) {
            this.biases[j] = isRelu ? 0.01 : 0.0;
        }
    }

    /**
     * Forward pass: z = W · x + b, a = activation(z)
     * @param {Float32Array | number[]} input 
     * @returns {Float32Array} Output activations
     */
    forward(input) {
        this.lastInput = input;

        for (let i = 0; i < this.outputDim; i++) {
            let sum = this.biases[i];
            const rowOffset = i * this.inputDim;
            for (let j = 0; j < this.inputDim; j++) {
                sum += this.weights[rowOffset + j] * input[j];
            }
            this.lastZ[i] = sum;

            // Activation
            if (this.activation === 'relu') {
                this.lastA[i] = sum > 0 ? sum : 0;
            } else {
                this.lastA[i] = sum; // Linear output
            }
        }

        return this.lastA;
    }

    /**
     * Backward pass: compute gradients and propagate delta to previous layer.
     * @param {Float32Array | number[]} gradOutput - dL/da from next layer
     * @returns {Float32Array} dL/d(input) to propagate to previous layer
     */
    backward(gradOutput) {
        const gradInput = new Float32Array(this.inputDim);

        // 1. Compute dL/dz through activation derivative
        const dL_dz = new Float32Array(this.outputDim);
        for (let i = 0; i < this.outputDim; i++) {
            if (this.activation === 'relu') {
                dL_dz[i] = this.lastZ[i] > 0 ? gradOutput[i] : 0;
            } else {
                dL_dz[i] = gradOutput[i];
            }
        }

        // 2. Accumulate weight and bias gradients
        for (let i = 0; i < this.outputDim; i++) {
            const delta = dL_dz[i];
            this.gradB[i] += delta;

            const rowOffset = i * this.inputDim;
            for (let j = 0; j < this.inputDim; j++) {
                this.gradW[rowOffset + j] += delta * this.lastInput[j];
                gradInput[j] += this.weights[rowOffset + j] * delta;
            }
        }

        return gradInput;
    }

    /**
     * Apply Adam optimizer step.
     * @param {number} learningRate 
     * @param {number} batchSize 
     */
    applyGradients(learningRate, batchSize = 1) {
        this.t += 1;
        const b1 = this.beta1;
        const b2 = this.beta2;
        const eps = this.epsilon;

        // Bias correction factors
        const corr1 = 1 - Math.pow(b1, this.t);
        const corr2 = 1 - Math.pow(b2, this.t);

        // Update weights
        for (let i = 0; i < this.weights.length; i++) {
            const g = this.gradW[i] / batchSize; // Average over batch
            this.mW[i] = b1 * this.mW[i] + (1 - b1) * g;
            this.vW[i] = b2 * this.vW[i] + (1 - b2) * g * g;

            const mHat = this.mW[i] / corr1;
            const vHat = this.vW[i] / corr2;

            this.weights[i] -= (learningRate / (Math.sqrt(vHat) + eps)) * mHat;
            this.gradW[i] = 0; // Reset accumulator
        }

        // Update biases
        for (let i = 0; i < this.biases.length; i++) {
            const g = this.gradB[i] / batchSize;
            this.mB[i] = b1 * this.mB[i] + (1 - b1) * g;
            this.vB[i] = b2 * this.vB[i] + (1 - b2) * g * g;

            const mHat = this.mB[i] / corr1;
            const vHat = this.vB[i] / corr2;

            this.biases[i] -= (learningRate / (Math.sqrt(vHat) + eps)) * mHat;
            this.gradB[i] = 0;
        }
    }

    /**
     * Copy weights and biases from another layer (for Target Network).
     * @param {DenseLayer} sourceLayer 
     */
    copyFrom(sourceLayer) {
        this.weights.set(sourceLayer.weights);
        this.biases.set(sourceLayer.biases);
    }
}

/**
 * Sequential Multi-Layer Perceptron.
 */
export class NeuralNetwork {
    /**
     * @param {number} inputDim - Dimension of input state vector
     * @param {number[]} hiddenDims - Array of neuron counts for hidden layers
     * @param {number} outputDim - Dimension of output action Q-values
     */
    constructor(inputDim, hiddenDims, outputDim) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.layers = [];

        let prevDim = inputDim;
        for (const hDim of hiddenDims) {
            this.layers.push(new DenseLayer(prevDim, hDim, 'relu'));
            prevDim = hDim;
        }
        // Output layer is linear for Q-values
        this.layers.push(new DenseLayer(prevDim, outputDim, 'linear'));
    }

    /**
     * Forward pass through all layers.
     * @param {number[] | Float32Array} inputVector 
     * @returns {Float32Array} Q-values for each action
     */
    predict(inputVector) {
        let current = inputVector;
        for (const layer of this.layers) {
            current = layer.forward(current);
        }
        return current;
    }

    /**
     * Perform one backpropagation step on a single sample.
     * @param {number[] | Float32Array} inputVector 
     * @param {number} actionIndex - Chosen action
     * @param {number} targetValue - TD target y = r + γ max Q(s', a')
     * @returns {number} Squared error loss
     */
    backwardSingle(inputVector, actionIndex, targetValue) {
        // Forward
        const output = this.predict(inputVector);
        const predictedQ = output[actionIndex];
        const error = predictedQ - targetValue; // dL/d(output) for MSE = 1/2 * (Q - y)^2

        // Output gradient: 0 for non-selected actions, error for the chosen action
        const gradOutput = new Float32Array(this.outputDim);
        gradOutput[actionIndex] = error;

        // Backprop through layers in reverse order
        let grad = gradOutput;
        for (let i = this.layers.length - 1; i >= 0; i--) {
            grad = this.layers[i].backward(grad);
        }

        return 0.5 * error * error;
    }

    /**
     * Apply accumulated gradients with Adam optimizer.
     * @param {number} learningRate 
     * @param {number} batchSize 
     */
    applyGradients(learningRate, batchSize = 1) {
        for (const layer of this.layers) {
            layer.applyGradients(learningRate, batchSize);
        }
    }

    /**
     * Copy all layer weights from another network.
     * @param {NeuralNetwork} targetNet 
     */
    copyWeightsFrom(otherNet) {
        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i].copyFrom(otherNet.layers[i]);
        }
    }

    /**
     * Serialize model to a plain object for JSON/localStorage.
     */
    toJSON() {
        return {
            inputDim: this.inputDim,
            outputDim: this.outputDim,
            layers: this.layers.map(l => ({
                inputDim: l.inputDim,
                outputDim: l.outputDim,
                activation: l.activation,
                weights: Array.from(l.weights),
                biases: Array.from(l.biases),
            }))
        };
    }

    /**
     * Restore weights from plain JSON object.
     * @param {Object} json 
     */
    fromJSON(json) {
        if (!json || !json.layers) return;
        for (let i = 0; i < Math.min(this.layers.length, json.layers.length); i++) {
            const src = json.layers[i];
            const dst = this.layers[i];
            if (src.weights && dst.weights.length === src.weights.length) {
                dst.weights.set(src.weights);
            }
            if (src.biases && dst.biases.length === src.biases.length) {
                dst.biases.set(src.biases);
            }
        }
    }
}
