// Sigmoid
function sig(x) {
  return 1 / (1 + Math.exp(-x));
}

// Derivative
function dSig(x) {
  return sig(x) * (1 - sig(x));
}

// Weights
let w1 = Math.random();
let w2 = Math.random();
let w3 = Math.random();
let w4 = Math.random();

let b1 = Math.random();
let b2 = Math.random();
let b3 = Math.random();

let lr = 0.05;
let gamma = 0.9;

// ===== GRID WORLD =====
let gridSize = 10;

function getState() {
  return {
    x: Math.floor(Math.random() * gridSize),
    y: Math.floor(Math.random() * gridSize)
  };
}

let target = getState();

// normalize state to 0–1
function encodeState(state, action) {
  return (
    state.x / gridSize +
    state.y / gridSize +
    action * 0.25
  );
}

// ===== NN =====
function NN(input) {
  let z1 = input * w1 + b1;
  let z2 = input * w2 + b2;

  let h1 = sig(z1);
  let h2 = sig(z2);

  let zOut = h1 * w3 + h2 * w4 + b3;
  let out = sig(zOut);

  return { out, h1, h2, z1, z2, zOut };
}

// ===== ACTIONS =====
// 0 = up, 1 = down, 2 = left, 3 = right
function getNextState(state, action) {
  let x = state.x;
  let y = state.y;

  if (action === 0) y--;
  if (action === 1) y++;
  if (action === 2) x--;
  if (action === 3) x++;

  x = Math.max(0, Math.min(gridSize - 1, x));
  y = Math.max(0, Math.min(gridSize - 1, y));

  return { x, y };
}

// ===== REWARD =====
function getReward(state) {
  let dist =
    Math.abs(state.x - target.x) +
    Math.abs(state.y - target.y);

  if (dist === 0) return 5;
  return -dist * 0.1; // closer = better
}

// ===== TRAIN =====
let state = getState();

for (let i = 0; i < 20000; i++) {

  // Q values
  let Q = [];
  for (let a = 0; a < 4; a++) {
    Q[a] = NN(encodeState(state, a)).out;
  }

  // epsilon-greedy
  let epsilon = 0.2;
  let action;

  if (Math.random() < epsilon) {
    action = Math.floor(Math.random() * 4);
  } else {
    action = Q.indexOf(Math.max(...Q));
  }

  let nextState = getNextState(state, action);
  let reward = getReward(nextState);

  // next Q
  let nextQ = [];
  for (let a = 0; a < 4; a++) {
    nextQ[a] = NN(encodeState(nextState, a)).out;
  }

  let maxNextQ = Math.max(...nextQ);

  // target
  let targetQ = reward + gamma * maxNextQ;

  // train chosen action
  let input = encodeState(state, action);
  let { out, h1, h2, z1, z2, zOut } = NN(input);

  let error = out - targetQ;
  let dOut = error * dSig(zOut);

  // backprop
  w3 -= lr * dOut * h1;
  w4 -= lr * dOut * h2;
  b3 -= lr * dOut;

  let dh1 = dOut * w3 * dSig(z1);
  let dh2 = dOut * w4 * dSig(z2);

  w1 -= lr * dh1 * input;
  w2 -= lr * dh2 * input;

  b1 -= lr * dh1;
  b2 -= lr * dh2;

  state = nextState;

  // reset if reached target
  if (state.x === target.x && state.y === target.y) {
    state = getState();
    target = getState();
  }
}

// ===== TEST =====
let testState = { x: 53, y: 20 };

let Q = [];
for (let a = 0; a < 4; a++) {
  Q[a] = NN(encodeState(testState, a)).out;
}

let action = Q.indexOf(Math.max(...Q));

console.log("State:", testState);
console.log("Q-values:", Q);
console.log("Best action:", action);
