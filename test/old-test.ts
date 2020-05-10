import Tensor from "../src/tensor";


console.log("----")

//console.log(images[2].toString())

function pool(input: Tensor, channel_first = false) {
    const ch = input.dim().d
    const [f_h, f_w] = [2,2]
    const [s_h, s_w] = [2,2]
    const patch_width = ((input.dim().c) - f_w)/s_w + 1
    const patch_height = ((input.dim().r) - f_h)/s_h + 1
    console.log(patch_width)
    let patch = new Tensor();
    if (channel_first) {
        patch.createEmptyArray(ch, patch_height, patch_width)
    } else {
        patch.createEmptyArray(patch_height, patch_width, ch)
    }

    for (let f = 0; f < ch; f++) {
        for (let r = 0; r < input.dim().r; r += s_h) {
            for (let c = 0; c < input.dim().c; c += s_w) {
                let val: number[] = []
                for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                    for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                        if (channel_first) {
                            val.push(input.get(f, r + c_f_h, c + c_f_w))
                        } else {
                            val.push(input.get(r + c_f_h, c + c_f_w, f))
                        }
                    }
                }
                if(channel_first) {
                    patch.set(f, r/s_h, c/s_w, Math.max(...val))
                } else {
                    patch.set(r/s_h, c/s_w, f, Math.max(...val))
                }
            }
        }
    }

    return patch
}

//const pooled = pool(images[2])
//console.log(pooled.toString())

function reverse_pool(gradients: Tensor, input: Tensor) {
    const [s_h,s_w] = [2,2]
    const [h, w, d] = input.shape()
    const [hh, ww] = gradients.shape()
    const [f_h, f_w] = [2,2]
    const t = new Tensor()
    t.createEmptyArray(h,w,d)
    for (let ch = 0; ch < d; ch++) {
        for (let r = 0; r < hh; r++) {
            for (let c = 0; c < ww; c++) {
                let i,j = 0
                const val = gradients.get(r, c, ch)
                for (let c_f_h = 0; c_f_h < f_h; c_f_h++) {
                    for (let c_f_w = 0; c_f_w < f_w; c_f_w++) {
                        if(input.get((r*s_h) + c_f_h, (c*s_w) + c_f_w, ch) == val) {
                            i = c_f_h
                            j = c_f_w
                            break
                        }
                    }
                }
                t.set((r*s_h) + i,(c*s_w) + j,ch, val)
            }
        }
    }
    return t
}

//console.log(reverse_pool(pooled, images[2]).toString())