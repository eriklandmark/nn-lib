import Model from "./lib/model";
import {GPU} from 'gpu.js';
import Matrix from "./lib/matrix";

const model = new Model([])

console.log(model.isGpuAvailable())

const gpu = new GPU({mode:"gpu"});

let size = 4;

const a = new Matrix()
a.createEmptyArray(size, size/2)
a.populateRandom()

const b = new Matrix()
b.createEmptyArray(size/2, size)
b.populateRandom()

const c = new Matrix()
c.createEmptyArray(a.dim().r, b.dim().c)
c.populateRandom()

const loss = gpu.createKernel(function(a, b) {
    return a[this.thread.x] - b[this.thread.x];
}).setOutput([a.dim().r, a.dim().c]);

const multiply = gpu.createKernel(function(a, b) {
    return a[this.thread.x] * b[this.thread.x];
}).setOutput([]);

//@ts-ignore
const superKernel = gpu.combineKernels(add, multiply, function(a, b, c) {
    return multiply(loss(a, b), c);
});

console.log(superKernel(a.toNumberArray(), b.toNumberArray(), c.toNumberArray()));


/*
const feedForwardKernel = gpu.createKernelMap({
    addResult: Matrix.addGpu(),
    multiplyResult: Matrix.mmGpu(),
    actvResult: Activations.sigmoid_gpu()
}, function(a, b, c) {
    //@ts-ignore
    return actv(add(mm(a, b), c[this.thread.y][this.thread.x]));
}, { output: [b.dim().c, a.dim().r], constants: {mmLength: a.dim().c}})
feedForwardKernel.setLoopMaxIterations(Math.max(a.dim().c, b.dim().r))


console.log(new Matrix(<Float32Array[]>feedForwardKernel(a.toNumberArray(), b.toNumberArray(), c.toNumberArray()).result).toString());

console.log(Activations.sigmoid((<Matrix>a.mm(b)).add(c)).toString())*/