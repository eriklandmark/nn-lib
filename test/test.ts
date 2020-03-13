import Tensor from "../src/tensor";

/*let dataset = new Dataset();

dataset.BATCH_SIZE = 1000
dataset.loadMnistTrain("./dataset/mnist", 1, false)
console.log(dataset.getBatch(0)[0].data.toString())
*/

const t = new Tensor([[[1], [7], [2], [1]], [[11], [1], [23], [8]], [[2], [2], [2], [7]]])
const filters = [
    new Tensor([[[1], [1]], [[0], [1]]]),
    new Tensor([[[1], [1]], [[1], [1]]]),
]

let shape = [4,5,7]
let sum = shape.reduce((acc, i) => acc * i)

for (let i = 0; i < sum; i++) {
    let r = Math.floor(i / (shape[1]*shape[2]))
    let c = Math.floor(i / (shape[2]) - (r*shape[1]))
    let d = Math.floor(i - (c*(shape[2])) - (r*shape[1]*shape[2]))
    console.log(r, c, d)
}

/*

let patch = new Tensor();
patch.createEmptyArray(patch_height, patch_width, nr_patches)
for (let d = 0; d < ch; d++) {
    for (let f = 0; f < nr_f; f++) {
        const b = (d * 3) + f
        for (let r = 0; r < patch_height; r++) {
            for (let c = 0; c < patch_width; c++) {
                let val = 0
                for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                    for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                        val += t.get(r + c_f_h, c + c_f_w, d) * fil.get(c_f_h, c_f_w, f)
                    }
                }
                patch.set(r,c,b, val)
            }
        }
    }
}



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