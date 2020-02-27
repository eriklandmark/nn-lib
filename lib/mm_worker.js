// Access the workerData by requiring it.
const { parentPort, workerData } = require("worker_threads");

// Main thread will pass the data you need
// through this event listener.
parentPort.on("message", (param) => {
    if (typeof param !== "number") {
        throw new Error("param must be a number.");
    }

    const {i, j} = param
    const {matrix, bMatrix} = workerData

    const result = matrix[i].reduce((acc, val, k) => acc + (val * bMatrix[k][j]), 0);

    // return the result to main thread.
    parentPort.postMessage({i: i, j: j, v: result});
});