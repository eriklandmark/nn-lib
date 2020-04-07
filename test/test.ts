import Tensor from "../src/tensor";
import {GPU} from "gpu.js"
import Sigmoid from "../src/lib/activations/sigmoid";
import Matrix from "../src/matrix";
import Vector from "../src/vector";
import Helper from "../src/helpers/helper";
import CrossEntropy from "../src/lib/losses/cross_entropy";
import Softmax from "../src/lib/activations/softmax";

let size = 300

const a = new Matrix()
a.createEmptyArray(size, size)
a.populateRandom()

const b = new Matrix()
b.createEmptyArray(size, size)
b.populateRandom()

const c = new Matrix()
c.createEmptyArray(a.dim().r, b.dim().c)
c.populateRandom()

const d = new Vector(size)
d.populateRandom()

const actv = new Sigmoid()

const gpu = new GPU({mode: "gpu"});
/*const filters = [
    new Tensor([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]),
    new Tensor([[[2], [3], [4]], [[5], [6], [7]], [[8], [9], [10]]]),
]

const images = [new Tensor([[[9, 54, 113], [139, 86, 118], [8,5,1]]]),
    new Tensor([[[1], [7], [24], [1]], [[113], [1], [23], [88]], [[2], [25], [62], [7]]])]*/
const channel_first = true

const channels = 3
const f_size = 5
const filters = []
/*filters.forEach((filter: Tensor) => {
    filter.createEmptyArray(f_size, f_size, channels);
    filter.populateRandom()
})*/

for (let i = 0; i < 3; i++) {
    const t = new Tensor()
    t.createEmptyArray(f_size, f_size, channels);
    t.populateRandom()
    filters.push(t)
}
filters.forEach((filter) => console.log(filter.toString()))

function gen(size: number) {
    const images = new Array(100).fill(new Tensor())
    images.forEach((image: Tensor) => {
        if (channel_first) {
            image.createEmptyArray(channels, size, size)
        } else {
            image.createEmptyArray(size, size, channels)
        }
        image.populateRandom()
    })

    return images
}

function conv(image: Tensor, patch_height: number, patch_width: number, filts: Tensor[], channel_first = true) {
    let patch = new Tensor();
    if (channel_first) {
        patch.createEmptyArray(filts.length, patch_height, patch_width)
    } else {
        patch.createEmptyArray(patch_height, patch_width, filts.length)
    }

    const chs = channel_first? image.dim().r : image.dim().d
    for (let f = 0; f < filts.length; f++) {
        for (let r = 0; r < patch_height; r++) {
            for (let c = 0; c < patch_width; c++) {
                let val: number = 0
                for (let c_f_c = 0; c_f_c < chs; c_f_c++) {
                    for (let c_f_h = 0; c_f_h < filts[f].dim().r; c_f_h++) {
                        for (let c_f_w = 0; c_f_w < filts[f].dim().c; c_f_w++) {
                            if (channel_first) {
                                val += image.get(c_f_c, r + c_f_h, c + c_f_w) * filts[f].get(c_f_h, c_f_w, c_f_c)
                            } else {
                                val += image.get(r + c_f_h, c + c_f_w, c_f_c) * filts[f].get(c_f_h, c_f_w, c_f_c)
                            }
                        }
                    }
                }
                if(channel_first) {
                    patch.set(f, r, c, val)
                } else {
                    patch.set(r, c, f, val)
                }
            }
        }
    }
    return patch
}

function buildKernel() {
    return gpu.createKernel(function (image, filter, ch_first) {
        let val: number = 0
        for (let c_f_c = 0; c_f_c < this.constants.channels; c_f_c++) {
            for (let c_f_h = 0; c_f_h < this.constants.filter_height; c_f_h++) {
                for (let c_f_w = 0; c_f_w < this.constants.filter_width; c_f_w++) {
                    if (ch_first) {
                        val += image[c_f_c][this.thread.y + c_f_h][this.thread.x + c_f_w] * filter[c_f_h][c_f_w][c_f_c]
                    } else {
                        val += image[this.thread.y + c_f_h][this.thread.x + c_f_w][c_f_c] * filter[c_f_h][c_f_w][c_f_c]
                    }
                }
            }
        }
        return val;
    }).setConstants({
        channels: channels,//channel_first? image.dim().r: image.dim().c,
        filter_height: f_size,
        filter_width: f_size,
    }).setPrecision("single")
}


const filterArray = filters.map((fil) => fil.toNumberArray())

let results = []

async function test() {
    for (let size = 10; size < 200; size += 10) {
        let res = {x: size, c_y: 0, g_y: 0}
        const images = gen(size)
        const imageArray = images.map((im: Tensor) => im.toNumberArray())
        const patch_width = ((size) - f_size + 1)
        const patch_height = ((size) - f_size + 1)
        const bp = buildKernel()
        bp.setOutput([patch_width, patch_height])
        bp.immutable = true
        const act = gpu.createKernel(actv.normal_gpu()).setOutput([patch_width, patch_height])
        act.immutable = true;
        let gpu_images = []
        let cpu_images = []

        res.g_y = await Helper.timeit(() => {
            gpu_images = imageArray.map((image) => {
                return filterArray.map((filter) => {return act(bp(image, filter, channel_first))})
            })
        }, false)

        res.c_y = await Helper.timeit(() => {
            cpu_images = images.map((image) => conv(image, patch_height, patch_width, filters))}, false)

        if (Math.abs(gpu_images[0][1][2][0] - cpu_images[0].get(1,2,0)) > 1000) {
            console.log("heheh")
        }
        //console.log(cpu_images[0].tensor)
        //console.log(gpu_images[0])
        console.log(res)
        results.push(res)
    }
}

let result = [
    { x: 10, c_y: 0.028, g_y: 0.457 },
    { x: 20, c_y: 0.043, g_y: 0.429 },
    { x: 30, c_y: 0.442, g_y: 0.45 },
    { x: 40, c_y: 0.803, g_y: 0.454 },
    { x: 50, c_y: 1.312, g_y: 0.536 },
    { x: 60, c_y: 1.912, g_y: 0.59 },
    { x: 70, c_y: 2.67, g_y: 0.726 },
    { x: 80, c_y: 3.52, g_y: 0.742 },
    { x: 90, c_y: 4.497, g_y: 0.814 },
    { x: 100, c_y: 5.628, g_y: 0.725 },
    { x: 110, c_y: 6.852, g_y: 0.869 },
    { x: 120, c_y: 8.227, g_y: 0.812 },
    { x: 130, c_y: 9.66, g_y: 0.968 },
    { x: 140, c_y: 11.361, g_y: 0.884 },
    { x: 150, c_y: 12.968, g_y: 1.003 },
    { x: 160, c_y: 14.787, g_y: 1.048 },
    { x: 170, c_y: 16.813, g_y: 1.164 },
    { x: 180, c_y: 18.904, g_y: 1.099 },
    { x: 190, c_y: 21.146, g_y: 1.255 },
    { x: 200, c_y: 23.355, g_y: 1.297 },
    { x: 210, c_y: 25.885, g_y: 1.273 },
    { x: 220, c_y: 28.632, g_y: 1.281 },
    { x: 230, c_y: 31.259, g_y: 1.37 },
    { x: 240, c_y: 34.154, g_y: 1.386 },
    { x: 250, c_y: 37.138, g_y: 1.533 },
    { x: 260, c_y: 40.44, g_y: 1.619 },
    { x: 270, c_y: 43.626, g_y: 1.732 },
    { x: 280, c_y: 47.167, g_y: 1.83 },
    { x: 290, c_y: 50.617, g_y: 1.924 },
    { x: 300, c_y: 54.558, g_y: 1.952 },
    { x: 310, c_y: 58.32, g_y: 2.079 },
    { x: 320, c_y: 62.461, g_y: 2.123 },
    { x: 330, c_y: 66.31, g_y: 2.228 },
    { x: 340, c_y: 70.925, g_y: 2.311 },
    { x: 350, c_y: 75.062, g_y: 2.383 },
    { x: 360, c_y: 80.01, g_y: 2.547 },
    { x: 370, c_y: 84.019, g_y: 2.522 },
    { x: 380, c_y: 89.729, g_y: 2.743 },
    { x: 390, c_y: 94.558, g_y: 2.799 },
    { x: 400, c_y: 99.558, g_y: 2.93 },
]

async function msa() {
    const a_f = (x: number) => x**2
    const b_f = (x: number) => x

    const a_vector = new Vector(results.map((res: any) => a_f(res.x)))
    const b_vector = new Vector(results.map((res: any) => b_f(res.x)))

    const y = new Vector(results.map((res) => res.g_y))

    const A = new Matrix([a_vector, b_vector])

    const VL = <Matrix> A.transpose().mm(A)
    const HL = A.transpose().mm(y)

    let xV = VL.inv()!.mm(HL)
    console.log(xV.toString())
}

async function run() {
    //await test()
    //await msa()
}


const patch_width = (4 - 3 + 1)
const patch_height = (4 - 3 + 1)

const filtr = [
    new Tensor([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]),
    new Tensor([[[2], [3], [4]], [[5], [6], [7]], [[8], [9], [10]]]),
]

const images = [new Tensor([[[9, 54, 113], [139, 86, 118], [8,5,1]]]),
    new Tensor([
        [[1], [7], [24], [1]],
        [[113], [1], [23], [88]],
        [[2], [25], [62], [7]],
        [[2], [25], [62], [7]]])]
console.log(conv(images[1], patch_height, patch_width, filtr, false).toString())





/*
import * as testAddon from "../build/Release/matrix_native.node";

let arr = new Float32Array(1)
arr[0] = 5

console.log(testAddon.mm([arr]))

/*let dataset = new Dataset();

dataset.BATCH_SIZE = 1000
dataset.loadMnistTrain("./dataset/mnist", 1, false)
console.log(dataset.getBatch(0)[0].data.toString())
*/

/*
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