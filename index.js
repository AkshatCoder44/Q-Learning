// Sigmoid
function sig(x) {
  return 1 / (1 + Math.exp(-x));
}

// Derivative
function dSig(x) {
  return sig(x) * (1 - sig(x));
}

// Weights (same)
let w1 = Math.random();
let w2 = Math.random();
let w3 = Math.random();
let w4 = Math.random();

let b1 = Math.random();
let b2 = Math.random();
let b3 = Math.random();

let lr = 0.05;
let gamma = 0.9;

// ===== NEURAL NETWORK =====
function NN(input) {
  let z1 = input * w1 + b1;
  let z2 = input * w2 + b2;

  let h1 = sig(z1);
  let h2 = sig(z2);

  let zOut = h1 * w3 + h2 * w4 + b3;
  let out = sig(zOut);

  return { out, h1, h2, z1, z2, zOut };
}

// ===== FAKE STATE =====
function getState() {
  return Math.random(); // later replace with snake info
}

function getNextState(state, action) {
  let change = (action - 1) * 0.1;
  return Math.min(1, Math.max(0, state + change));
}

// ===== REWARD =====
function getReward(state) {
  if (state > 0.8) return 1;
  if (state < 0.2) return -1;
  return -0.1;
}

// ===== TRAIN =====
let state = getState();

for (let i = 0; i < 10000; i++) {

  // ===== Q VALUES FOR 3 ACTIONS =====
  let Q0 = NN(state + 0).out; // action 0
  let Q1 = NN(state + 1).out; // action 1
  let Q2 = NN(state + 2).out; // action 2

  // ===== PICK BEST ACTION =====
  let action = 0;
  let maxQ = Q0;
  let epsilon = 0.2;
  if (Math.random() < epsilon) {
    action = Math.floor(Math.random() * 3); // random
  } else {
    action = 0;
    let maxQ = Q0;
  
    if (Q1 > maxQ) {
      maxQ = Q1;
      action = 1;
    }
    if (Q2 > maxQ) {
      maxQ = Q2;
      action = 2;
    }
  }

  // ===== TAKE ACTION → NEXT STATE =====
  let nextState = getNextState(state, action);
  let reward = getReward(nextState);

  // ===== NEXT Q VALUES =====
  let nextQ0 = NN(nextState + 0).out;
  let nextQ1 = NN(nextState + 1).out;
  let nextQ2 = NN(nextState + 2).out;

  let maxNextQ = Math.max(nextQ0, nextQ1, nextQ2);

  // ===== TARGET (Q LEARNING) =====
  let target = reward + gamma * maxNextQ;

  // ===== TRAIN ONLY CHOSEN ACTION =====
  let input = state + action;

  let { out, h1, h2, z1, z2, zOut } = NN(input);

  let error = out - target;
  let dOut = error * dSig(zOut);

  // backprop (same as yours)
  w3 -= lr * dOut * h1;
  w4 -= lr * dOut * h2;
  b3 -= lr * dOut;

  let dh1 = dOut * w3 * dSig(z1);
  let dh2 = dOut * w4 * dSig(z2);

  w1 -= lr * dh1 * input;
  w2 -= lr * dh2 * input;

  b1 -= lr * dh1;
  b2 -= lr * dh2;
  
  // move forward
  state = nextState;
}

// ===== TEST WITH A REAL STATE =====
let testState = 1; // try different values

// get Q-values
let Q0 = NN(testState + 0).out;
let Q1 = NN(testState + 1).out;
let Q2 = NN(testState + 2).out;

// pick best action
let action = 0;
let maxQ = Q0;

if (Q1 > maxQ) {
  maxQ = Q1;
  action = 1;
}
if (Q2 > maxQ) {
  maxQ = Q2;
  action = 2;
}

console.log("State:", testState);
console.log("Q-values:", Q0, Q1, Q2);
console.log("Chosen action:", action);
