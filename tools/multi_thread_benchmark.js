const cp = require("child_process")
const os = require("os")

function createNode() {
    return new Promise((resolve => {
        const proc = cp.fork("./tools/benchmark_worker.js")
        proc.send("run")
        proc.on('message', (data) => {
            resolve(data)
        });
    }))
}

async function run() {
    const threads = os.cpus().length

    console.log("Starting benchmarking...")
    console.log("Using", threads, "number of threads")

    const processes = []
    for (let i = 0; i < threads; i++) {
        processes.push(createNode())
    }

    const startTime = Date.now()
    const results = await Promise.all(processes)
    const duration = (Date.now() - startTime) / 1000.0

    const score = results.reduce((acc, v) => acc + v, 0)
    console.log("Total score:", score, "points.", "Total time", duration, "seconds.")
}

run()
