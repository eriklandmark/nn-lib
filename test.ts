import Model from "./lib/model";
import {GPU} from 'gpu.js';
import Matrix from "./lib/matrix";

const model = new Model([])

console.log(model.isGpuAvailable())

const gpu = new GPU();

const a = new Matrix()
a.createEmptyArray(1000, 1000)
a.populateRandom()

const b = new Matrix()
b.createEmptyArray(1000, 1000)
b.populateRandom()

const multiplyMatrix = gpu.createKernel(function(a, b, length) {
    let sum = 0;
    for (let i = 0; i < length; i++) {
        sum += a[this.thread.y][i] * b[i][this.thread.x];
    }
    return sum;
}).setOutput([1000, 1000]);

const c = multiplyMatrix(a.toNumberArray(), b.toNumberArray(), a.dim().c);

const d = new Matrix([[1,8,2], [1,2,3]])

console.log(d.sum(0).toString())