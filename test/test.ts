import Model from "../lib/model";

const model = new Model([])

console.log(model.isGpuAvailable())
/*
const gpu = new GPU({mode:"gpu"});

let size = 1000;

const a = new Matrix()
a.createEmptyArray(size, size)
a.populateRandom()

const b = new Matrix()
b.createEmptyArray(size, size)
b.populateRandom()

const c = new Matrix()
c.createEmptyArray(a.dim().r, b.dim().c)
c.populateRandom()

const add = gpu.createKernel(Matrix.addGpu()).setOutput([a.dim().r, a.dim().c]);
const multiply = gpu.createKernel(Matrix.mmGpu()).setOutput([a.dim().r, b.dim().c]).setConstants({mmLength: a.dim().c});

//@ts-ignore
const superKernel = gpu.combineKernels(...[add, multiply], function(a, b) {
    return multiply(multiply(add(a, b), b), b);
})

const feedForwardKernal = gpu.createKernel(function(a, b, c) {
    //@ts-ignore
    return actv(add(mm(a, b), c[this.thread.y][this.thread.x]));
}).setOutput([b.dim().c, a.dim().r]).setFunctions([Matrix.addGpu(), Matrix.mmGpu(), Activations.sigmoid_gpu()]).setConstants({mmLength: a.dim().c});


console.log(new Matrix(feedForwardKernal(a.toNumberArray(), b.toNumberArray(), c.toNumberArray()) as Float32Array[]).toString());
console.log((<Matrix>Activations.sigmoid((<Matrix>a.mm(b)).add(c))).toString())


//console.log(new Matrix(superKernel(a.toNumberArray(), b.toNumberArray())).toString());
//console.log((<Matrix>a.add(b).mm(b)).mm(b).toString())
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

//@ts-ignore
const omegaKernal = gpu.combineKernels(multiply, add, function(a, b, c) {
    return multiply(add(a, b), b);
});

console.log(new Matrix(omegaKernal(a.toNumberArray(), b.toNumberArray(), c.toNumberArray()) as Float32Array[]).toString());


//console.log(new Matrix(<Float32Array[]>feedForwardKernel(a.toNumberArray(), b.toNumberArray(), c.toNumberArray()).result).toString());

//console.log(Activations.sigmoid((<Matrix>a.mm(b)).add(c)).toString())*/