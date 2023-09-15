export default class Tensor {

    t: Float64Array | Float64Array[] | Float64Array[][] | Float64Array[][][] = []
    shape: number[] = []
    dim: number = 0

    constructor(v: any[] | Float64Array = [], shape: boolean = false) {
        if (shape) {
            this.shape = <number[]>v
            this.createFromShape(this.shape)
            this.dim = this.shape.length
        } else {
            if (v.length == 0) {
                this.shape = [0]
            } else {
                this.calculateShape(v)
                this.createTensor(v)
            }
        }
    }

    public static createIdentityMatrix(n: number) {
        const t = new Tensor([n,n], true)
        for (let i = 0; i < n; i++) {
            t.t[i][i] = 1
        }
        return t
    }

    private createTensor(v: any[] | Float64Array) {
        if (v instanceof Float64Array) {
            this.t = v
        } else if (typeof v[0] == "number") {
            this.t = Float64Array.from(v);
        } else {
            if (this.shape.length == 2) {
                this.t = <Float64Array[]>[]
                for (let i = 0; i < this.shape[0]; i++) {
                    this.t.push(Float64Array.from(v[i]))
                }
            } else if (this.shape.length == 3) {
                this.t = <Float64Array[][]>[]
                for (let i = 0; i < this.shape[0]; i++) {
                    this.t.push([]);
                    for (let j = 0; j < this.shape[1]; j++) {
                        this.t[i].push(Float64Array.from(v[i][j]))
                    }
                }
            } else if (this.shape.length == 4) {
                this.t = <Float64Array[][][]>[]
                for (let i = 0; i < this.shape[0]; i++) {
                    this.t.push([]);
                    for (let j = 0; j < this.shape[1]; j++) {
                        this.t[i].push([]);
                        for (let k = 0; k < this.shape[2]; k++) {
                            this.t[i][j].push(Float64Array.from(v[i][j][k]))
                        }
                    }
                }
            }
        }
    }

    private createFromShape(shape: number[]): void {
        this.shape = shape
        if (shape.length == 1) {
            this.t = new Array(shape[0]).fill(0);
        } else if (shape.length == 2) {
            this.t = <Float64Array[]>[]
            for (let i = 0; i < shape[0]; i++) {
                this.t.push(new Float64Array(shape[1]).fill(0))
            }
        } else if (shape.length == 3) {
            this.t = <Float64Array[][]>[]
            for (let i = 0; i < shape[0]; i++) {
                this.t.push([]);
                for (let j = 0; j < shape[1]; j++) {
                    this.t[i].push(new Float64Array(shape[2]).fill(0))
                }
            }
        } else if (shape.length == 4) {
            this.t = <Float64Array[][][]>[]
            for (let i = 0; i < shape[0]; i++) {
                this.t.push([]);
                for (let j = 0; j < shape[1]; j++) {
                    this.t[i].push([]);
                    for (let k = 0; k < shape[2]; k++) {
                        this.t[i][j].push(new Float64Array(shape[3]).fill(0))
                    }
                }
            }
        }
    }

    private calculateShape(arr: any) {
        let shape = []
        const f = (v: any) => {
            shape.push(v.length)
            if (!(v instanceof Float64Array || typeof v[0] == "number")) {
                f(v[0])
            }
        }

        if (arr instanceof Float64Array || typeof arr[0] == "number") {
            this.shape = [arr.length]
        } else {
            shape.push(arr.length)
            f(arr[0])
            this.shape = shape
        }
        this.dim = this.shape.length
    }

    public get(pos: number[]): number {
        if (pos.length == 1) {
            if (!isFinite(<number>this.t[pos[0]])) {
                console.trace()
                throw "Getting an NaN... (" + this.t[pos[0]] + ")"
            }
            return <number>this.t[pos[0]]
        } else if (pos.length == 2) {
            if (!isFinite(this.t[pos[0]][pos[1]])) {
                console.trace()
                throw "Getting an NaN... (" + this.t[pos[0]][pos[1]] + ")"
            }
            return this.t[pos[0]][pos[1]]
        } else if (pos.length == 3) {
            if (!isFinite(this.t[pos[0]][pos[1]][pos[2]])) {
                console.trace()
                throw "Getting an NaN... (" + this.t[pos[0]][pos[1]][pos[2]] + ")"
            }
            return this.t[pos[0]][pos[1]][pos[2]]
        } else if (pos.length == 4) {
            if (!isFinite(this.t[pos[0]][pos[1]][pos[2]][pos[3]])) {
                console.trace()
                throw "Getting an NaN... (" + this.t[pos[0]][pos[1]][pos[2]][pos[3]] + ")"
            }
            return this.t[pos[0]][pos[1]][pos[2]][pos[3]]
        }

    }

    getLinearPos(pos: number[]): number {
        if (pos.length == 1) {
            return pos[0]
        } else if (pos.length == 2) {
            return pos[0] * this.shape[1] + pos[1]
        } else if (pos.length == 3) {
            return pos[0] * this.shape[1] * this.shape[2] + pos[1] * this.shape[2] + pos[2]
        } else if (pos.length == 4) {
            return pos[0] * this.shape[1] * this.shape[2] * this.shape[3] + pos[1] * this.shape[2] * this.shape[3] + pos[2] * this.shape[3] + pos[3]
        }
    }

    public set(pos: number[], v: number): void {
        if (!isFinite(v) || isNaN(v)) {
            console.trace()
            throw "Number is NaN... (" + v + ")"
        }
        if (this.dim == 1) {
            this.t[pos[0]] = v
        } else if (this.dim == 2) {
            this.t[pos[0]][pos[1]] = v
        } else if (this.dim == 3) {
            this.t[pos[0]][pos[1]][pos[2]] = v
        } else if (this.dim == 4) {
            this.t[pos[0]][pos[1]][pos[2]][pos[3]] = v
        }
    }

    public count(): number {
        return this.shape.reduce((acc: number, d: number) => acc * d, 1)
    };

    public static toCategorical(index: number, size: number) {
        const v = new Tensor([size], true);
        v.t[index] = 1
        return v
    }

    public static fromJsonObject(obj: any[][]): Tensor {
        if (obj.length == 0) {
            return new Tensor()
        } else if (typeof obj["0"] == "number") {
            return new Tensor(Object.keys(obj).map(
                (_item: string, index) => {
                    return obj[index.toString()]
                }
            ))
        } else if (typeof obj["0"]["0"] == "number") {
            return new Tensor(obj.map((row: any) => {
                return Object.keys(row).map((item) => row[item])
            }))
        } else if (typeof obj["0"]["0"]["0"] == "number") {
            return new Tensor(obj.map((row: any[]) => {
                return row.map((col: any) => {
                    return Object.keys(col).map((item) => col[item])
                })
            }))
        } else if (typeof obj["0"]["0"]["0"]["0"] == "number") {
            return new Tensor(obj.map((row: any[]) => {
                return row.map((col: any) => {
                    return col.map((depth: any) => {
                        return Object.keys(depth).map((item) => depth[item])
                    })
                })
            }))
        }
    }

    public equalShape(t: Tensor): boolean {
        if (this.dim !== t.dim)
            return false

        for (let i = 0; i < this.dim; i++) {
            if (this.shape[i] !== t.shape[i]) {
                return false
            }
        }

        return true
    }

    public equal(t: Tensor): boolean {
        if(!this.equalShape(t)) {
            return false
        }

        let ans = true

        this.iterate((pos: number[]) => {
            if (Math.abs(this.get(pos) - t.get(pos)) > Number.EPSILON) {
                ans = false
            }
        }, true)

        return ans
    }

    public toNumberArray(): any[] {
        if (this.dim == 1) {
            return [].slice.call(this.t)
        } else if (this.dim == 2) {
            return (<Float64Array[]>this.t).map((floatArray) => [].slice.call(floatArray))
        } else if (this.dim == 3) {
            return (<Float64Array[][]>this.t).map((array) =>
                array.map((floatArray) => [].slice.call(floatArray)))
        }
    }

    public iterate(func: Function, use_pos: boolean = false, channel_first = false): void {
        if (this.dim == 1) {
            for (let i: number = 0; i < this.shape[0]; i++) {
                if (use_pos) {
                    func([i])
                } else {
                    func(i)
                }
            }
        } else if (this.dim == 2) {
            for (let i: number = 0; i < this.shape[0]; i++) {
                for (let j: number = 0; j < this.shape[1]; j++) {
                    if (use_pos) {
                        func([i, j])
                    } else {
                        func(i, j)
                    }
                }
            }
        } else if (this.dim == 3) {
            if (channel_first) {
                for (let k: number = 0; k < this.shape[2]; k++) {
                    for (let i: number = 0; i < this.shape[0]; i++) {
                        for (let j: number = 0; j < this.shape[1]; j++) {
                            if (use_pos) {
                                func([i, j, k])
                            } else {
                                func(i, j, k)
                            }
                        }
                    }
                }
            } else {
                for (let i: number = 0; i < this.shape[0]; i++) {
                    for (let j: number = 0; j < this.shape[1]; j++) {
                        for (let k: number = 0; k < this.shape[2]; k++) {
                            if (use_pos) {
                                func([i, j, k])
                            } else {
                                func(i, j, k)
                            }
                        }
                    }
                }
            }
        } else if (this.dim == 4) {
            for (let i: number = 0; i < this.shape[0]; i++) {
                for (let j: number = 0; j < this.shape[1]; j++) {
                    for (let k: number = 0; k < this.shape[2]; k++) {
                        for (let l: number = 0; l < this.shape[3]; l++) {
                            if (use_pos) {
                                func([i, j, k, l])
                            } else {
                                func(i, j, k, l)
                            }
                        }
                    }
                }
            }
        }
    }

    numberToString(nr: number, precision: number = 5, autoFill: boolean = false): string {
        const expStr = nr.toExponential()
        return (+expStr.substring(0, expStr.lastIndexOf("e"))).toPrecision(precision)
            + expStr.substring(expStr.lastIndexOf("e")) +
            (autoFill ? " ".repeat(4 - expStr.substring(expStr.lastIndexOf("e")).length) : "")
    }

    public toString(max_rows: number = 10, precision: number = 3): string {
        if (this.dim == 0) {
            return "Tensor: []"
        } else if (this.dim == 1) {
            return (<Float64Array>this.t).reduce((acc, v) => {
                acc += `    ${this.numberToString(v, precision, true)}\n`
                return acc;
            }, `Tensor: ${this.shape[0]} [\n`) + " ]"
        } else if (this.dim == 2) {
            return (<any[]>this.t).slice(0, Math.min(max_rows, this.t.length)).reduce((acc, i) => {
                acc += i.slice(0, Math.min(max_rows, i.length)).reduce((s: string, i: number) => {
                    s += " "//.repeat(Math.max(maxCharCount - i.toPrecision(precision).length, 1))
                    s += this.numberToString(i, precision, true);
                    return s;
                }, "    ")
                acc += i.length > max_rows ? "  ... +" + (i.length - max_rows) + " elements\n" : "\n"
                return acc;
            }, `Tensor: ${this.shape[0]}x${this.shape[1]} [\n`) + (this.t.length > max_rows ?
                "    ... +" + (this.t.length - max_rows) + " rows \n]" : " ]")
        } else if (this.dim == 3) {
            let maxCharCount = 0;
            this.iterate((i: number, j: number, k: number) => {
                let val = this.t[i][j][k].toString()
                if (val.length > maxCharCount) maxCharCount = val.length
            })
            maxCharCount = Math.min(maxCharCount, 7)
            let string = `Tensor: ${this.shape[0]}x${this.shape[1]}x${this.shape[2]} [\n`
            for (let d = 0; d < this.shape[2]; d++) {
                string += (<any[]>this.t).slice(0, Math.min(max_rows, this.t.length)).reduce((acc, i) => {
                    acc += i.slice(0, Math.min(max_rows, i.length)).reduce((s: string, j: number) => {
                        const v = this.numberToString(j[d], precision, true)
                        s += " "//.repeat(Math.max(maxCharCount - j[d].toString().length, 0))
                        s += v //j[d].toString().substr(0, maxCharCount) + " ";
                        return s;
                    }, "    ")
                    acc += i.length > max_rows ? " ... +" + (i.length - max_rows) + " elements\n" : "\n"
                    return acc;
                }, "") + (this.t.length > max_rows ?
                    "    ... +" + (this.t.length - max_rows) + " rows \n" : "\n")
            }
            return string + "]"
        }
    }

    public print(max_rows: number = 10, precision: number = 3): void {
        console.log(this.toString(max_rows, precision))
    }

    public copy(full: boolean = true): Tensor {
        if (this.shape[0] == 0 && [this.shape.length == 1]) {
            return new Tensor()
        } else {
            let t = new Tensor(this.shape, true)
            if (full) {
                t.iterate((pos: number[]) => t.set(pos, this.get(pos)), true)
            }
            return t
        }
    }

    public populateRandom(seed: number | null = null) {
        this.iterate((pos: number[]) => {
            if (seed) {
                const x = Math.sin(seed + this.getLinearPos(pos)) * 10000;
                this.set(pos, x - Math.floor(x))
            } else {
                this.set(pos, Math.random() * 2 - 1)
            }

        }, true)
    }

    public empty(): boolean {
        return this.shape[0] == 0 || this.shape[1] == 0 || this.shape[2] == 0 || this.shape[3] == 0
    }

    public vectorize(channel_first: boolean = false): Tensor {
        const t = new Tensor([this.count()], true)
        let index = 0;
        if (this.dim == 1) {
            return this.copy(true)
        } else {
            this.iterate((pos: number[]) => {
                t.t[index] = this.get(pos)
                index += 1
            }, true)
        }

        return t
    }

    public div(v: number | Tensor, safe:boolean = false): Tensor {
        let t = this.copy(false);
        if (v instanceof Tensor) {
            if (!this.equalShape(v)) {
                console.trace()
                throw "Tensor Division: Not the same shape"
            }
            this.iterate((pos: number[]) => {
                const n: number = safe? v.get(pos) + 10**-7 : v.get(pos)
                t.set(pos, this.get(pos) / n)
            }, true);
        } else {
            const n: number = safe? v + 10**-7 : v
            this.iterate((pos: number[]) => {
                t.set(pos, this.get(pos) / n)
            }, true);
        }
        return t
    }

    public mul(v: number | Tensor): Tensor {
        let t = this.copy(false);
        if (v instanceof Tensor) {
            if (!this.equalShape(v)) {
                console.trace()
                throw "Tensor Multiplication: Not the same shape"
            }
            this.iterate((pos: number[]) => {
                t.set(pos, this.get(pos) * v.get(pos))
            }, true);
        } else {
            this.iterate((pos: number[]) => {
                t.set(pos, this.get(pos) * v)
            }, true);
        }
        return t
    }

    public sub(v: number | Tensor): Tensor {
        let t = this.copy(false);
        if (v instanceof Tensor) {
            if (!this.equalShape(v)) {
                console.trace()
                throw "Tensor Subtraction: Not the same shape"
            }
            this.iterate((pos: number[]) => {
                t.set(pos, this.get(pos) - v.get(pos))
            }, true);
        } else {
            this.iterate((pos: number[]) => {
                t.set(pos, this.get(pos) - v)
            }, true);
        }
        return t
    }

    public add(v: number | Tensor): Tensor {
        let t = this.copy(false);
        if (v instanceof Tensor) {
            if (!this.equalShape(v)) {
                console.trace()
                throw "Tensor Addition: Not the same shape"
            }
            this.iterate((pos: number[]) => {
                t.set(pos, this.get(pos) + v.get(pos))
            }, true);
        } else {
            this.iterate((pos: number[]) => {
                t.set(pos, this.get(pos) + v)
            }, true);
        }
        return t
    }

    public pow(v: number): Tensor {
        let t = this.copy(false);
        this.iterate((pos: number[]) => {
            t.set(pos, this.get(pos) ** v)
        }, true);
        return t
    }

    public sqrt(): Tensor {
        let t = this.copy(false);
        this.iterate((pos: number[]) => {
            t.set(pos, Math.sqrt(this.get(pos)))
        }, true);
        return t
    }

    public exp(base: null | number = null): Tensor {
        let t = this.copy(false);
        this.iterate((pos: number[]) => {
            t.set(pos, base ? base ** this.get(pos) : Math.exp(this.get(pos)))
        }, true);
        return t
    }

    public inv_el(eps = 10 ** -7): Tensor {
        let t = this.copy(false);
        this.iterate((pos: number[]) => {
            t.set(pos, this.get(pos) == 0 ? 1 / (this.get(pos) + eps) : 1 / this.get(pos))
        }, true);
        return t
    }

    public fill(scalar: number): Tensor {
        let t = this.copy(false);
        this.iterate((pos: number[]) => {
            t.set(pos, scalar)
        }, true);
        return t
    }

    public log(): Tensor {
        let t = this.copy(false);
        this.iterate((pos: number[]) => {
            t.set(pos, Math.log(this.get(pos)))
        }, true);
        return t
    }

    public dot(b: Tensor): Tensor {
        if (this.dim == 2) {
            if (b.dim == 1) {
                if (b.shape[0] != this.shape[1]) {
                    console.trace()
                    throw "Matrix Multiplication (Vector): Wrong dimension..\n" +
                    "This: [ " + this.shape[0] + " , " + this.shape[1] + " ] | Other: [ " + b.shape[0] + " ]"
                }

                const t = new Tensor([this.shape[1]], true)
                for (let i = 0; i < this.shape[1]; i++) {
                    for (let j = 0; j < this.shape[0]; j++) {
                        (<Float64Array> t.t)[i] += (<Float64Array> b.t)[j] * this.t[i][j]
                    }
                }
                return t;
            } else if (b.dim == 2) {
                if (this.shape[1] != b.shape[0]) {
                    console.trace()
                    throw "Matrix Multiplication (Matrix): Wrong dimension..\n" +
                    "This: [ " + this.shape[0] + " , " + this.shape[1] + " ] | Other: [ " + b.shape[0] + " , " + b.shape[1] + " ]"
                }

                const t = new Tensor([this.shape[0], b.shape[1]], true)

                for (let i: number = 0; i < this.shape[0]; i++) {
                    for (let j: number = 0; j < this.shape[1]; j++) {
                        for (let k: number = 0; k < this.shape[1]; k++) {
                            t.t[i][j] += (<Float64Array[]>this.t)[i][k] * b.t[k][j]
                        }
                    }
                }
                return t
            }
        } else {
            console.trace()
            throw "Dot Multiplication: Shape must be of size 2 (Matrix)..\n"
        }
    }

    public padding(padding_height: number, padding_width: number, axis: number[] = [0, 1]) {
        if (axis[0] == 0 && axis[1] == 1) {
            const t = new Tensor([2 * padding_height + this.shape[0], 2 * padding_width + this.shape[1], this.shape[3]], true)

            for (let i = 0; i < this.shape[0]; i++) {
                for (let j = 0; j < this.shape[1]; j++) {
                    for (let c = 0; c < this.shape[2]; c++) {
                        t.set([i + padding_height, j + padding_width, c], this.get([i, j, c]))
                    }
                }
            }
            return t
        }
    }

    public im2patches(patch_height: number, patch_width: number, filter_height: number, filter_width: number): Tensor {
        const cols = []
        for (let r = 0; r < patch_height; r++) {
            for (let c = 0; c < patch_width; c++) {
                const v = []
                for (let c_f_c = 0; c_f_c < this.shape[2]; c_f_c++) {
                    for (let c_f_h = 0; c_f_h < filter_height; c_f_h++) {
                        for (let c_f_w = 0; c_f_w < filter_width; c_f_w++) {
                            v.push(this.get([r + c_f_h, c + c_f_w, c_f_c]))
                        }
                    }
                }
                cols.push(v)
            }
        }

        return new Tensor(cols)
    }

    public rotate180(): Tensor {
        const t = this.copy(false)
        if (this.dim == 4) {
            this.iterate((n: number, i: number, j: number, k: number) => {
                t.t[n][this.shape[1] - 1 - i][this.shape[2] - 1 - j][k] = this.get([n, i, j, k])
            })
        }
        return t
    }

    public rowVectors() {
        if (this.dim == 2) {
            return (<Float64Array[]>this.t).map((row) => new Tensor(row))
        }
    }

    public argmax(index = -1, axis: number = 0): number {
        if (this.dim == 1) {
            return (<Float64Array>this.t).reduce((acc: number, va: number, ind) =>
                va > <number> this.t[acc] ? ind : acc, 0)
        } else if (this.dim == 2) {
            if (axis == 0) {
                if (index < 0) {
                    return 0;
                } else {
                    return (<Float64Array[]>this.t)[index].reduce((acc: number, va: number, ind) =>
                        va > this.t[index][acc] ? ind : acc, 0)
                }
            } else {
                if (index < 0) {
                    return 0;
                } else {
                    let maxIndex = 0;
                    for (let j = 0; j < this.shape[0]; j++) {
                        if (Math.abs(this.t[j][index]) > Math.abs(this.t[maxIndex][index])) {
                            maxIndex = j;
                        }
                    }
                    return maxIndex;
                }
            }
        }
    }

    public reshape(shape: number[]): Tensor {
        if (this.dim == 1) {
            if (this.count() != shape.reduce((acc, n) => acc * n, 1)) {
                throw "Product of shape must be the same as size of vector!"
            }
            const t = new Tensor(shape, true);
            let [_h, w, d] = shape

            this.iterate((val: number, i: number) => {
                const r = Math.floor(i / (w * d))
                const c = Math.floor(i / (d) - (r * w))
                const g = Math.floor(i - (c * d) - (r * w * d))
                t.t[r][c][g] = val
            })
            return t
        }
    }

    public sum(axis: number = -1, keepDims = false): number | Tensor {
        if (this.dim == 1) {
            return (<Float64Array>this.t).reduce((acc, val) => acc + val);
        } else if (this.dim == 2) {
            if (keepDims) {
                let t = this.copy();
                if (axis == 1) {
                    for (let i = 0; i < t.shape[0]; i++) {
                        const sum = (<Float64Array> t.t[i]).reduce((acc, val) => acc + val, 0);
                        (<Float64Array> t.t[i]).forEach((_val, j) => t.t[i][j] = sum)
                    }
                } else if (axis == 0) {
                    for (let j = 0; j < this.shape[1]; j++) {
                        let sum = 0;
                        for (let i = 0; i < this.shape[0]; i++) {
                            sum += this.t[i][j]
                        }
                        for (let i = 0; i < this.shape[0]; i++) {
                            t.t[i][j] = sum
                        }
                    }
                } else if (axis == -1) {
                    const sum = (<Float64Array[]>t.t).reduce((acc, val) => {
                        acc += val.reduce((acc, val) => acc + val, 0)
                        return acc;
                    }, 0);
                    this.iterate((i: number, j: number) => {
                        t.t[i][j] = sum
                    });
                } else if (axis >= 2) {
                    return this.copy()
                }
                return t
            } else {
                if (axis == -1) {
                    return (<Float64Array[]>this.t).reduce((acc, val) => {
                        acc += val.reduce((acc, val) => acc + val, 0)
                        return acc;
                    }, 0);
                } else if (axis == 0) {
                    let t = new Tensor([1, this.shape[1]], true)
                    this.iterate((i: number, j: number) => {
                        t.t[0][j] = this.get([i, j]) + t.get([0, j])
                    })
                    return t;
                } else if (axis == 1) {
                    let t = new Tensor([this.shape[0], 1], true)
                    this.t.forEach((arr, i) => {
                        t.t[i][0] = arr.reduce((acc: number, val: number) => acc + val, 0);
                    });
                    return t;
                } else if (axis == 2) {
                    return this.copy()
                }
                return 0
            }
        }
    }

    public norm(p:number = 2) {
        let acc = 0
        this.iterate((pos) => {acc += this.get(pos)**p}, true)
        return acc**(1/p)
    }

    public mean(axis = -1, keep_dims = false): number | Tensor {
        if (this.dim == 1) {
            return <number>this.sum() / this.count();
        } else if (this.dim == 2) {
            if (axis == -1) {
                return <number>this.sum(-1, false) / this.count()
            } else if (axis == 0 || axis == 1) {
                return (<Tensor>this.sum(axis, keep_dims)).div(axis == 0 ? this.shape[0] : this.shape[1])
            }
        }
    }

    public repeat(axis = 0, times = 1) {
        if (this.dim == 2) {
            if (axis == 0) {
                const t = new Tensor([times, this.shape[1]], true)
                //t.t.fill(this.t[0])
                return t
            }
        }

    }

    public transpose(): Tensor {
        if (this.dim == 1) {
            let t = new Tensor([this.shape[0], 1], true)
            this.iterate((i: number) => {
                t.t[i][0] = this.t[i]
            });
            return t
        } else {
            let t = new Tensor([this.shape[1], this.shape[0]], true)
            this.iterate((i: number, j: number) => {
                t.t[j][i] = this.t[i][j]
            });
            return t;
        }
    }

    public trace(): number {
        if (this.dim == 2) {
            if (this.shape[0] == this.shape[1]) {
                let sum = 0
                for (let i = 0; i < this.shape[0]; i++) {
                    sum += this.t[i][i]
                }
                return sum
            } else {
                throw "trace(): Tensor must be a square"
            }
        } else {
            throw "trace(): Tensor must be an matrix."
        }
    }

    public isUpperTriangular() {
        const r = Math.min(this.shape[0], this.shape[1])
        for (let i = 1; i < r; i++) {
            for (let j = 0; j < i; j++) {
                if (this.get([i, j]) != 0) {
                    return false
                }
            }
        }
        return true
    }


    public isLowerTriangular() {
        const r = Math.min(this.shape[0], this.shape[1]) - 1
        for (let i = 0; i < r; i++) {
            for (let j = this.shape[1] - 1; j >= i + 1; j--) {
                if (this.get([i, j]) != 0) {
                    return false
                }
            }
        }
        return true
    }

    public isSymmetric() {
        return this.equal(this.transpose())
    }

    public isDiagonal() {
        return this.isUpperTriangular() && this.isLowerTriangular()
    }

    public rref(verify: boolean = true) {
        const t = this.copy(true)

        if (!verify || (!this.isUpperTriangular() && !this.isLowerTriangular())) {
            let h = 0
            let k = 0

            while (h < t.shape[0] && k < t.shape[1]) {
                let i_max = h
                for (let i = h; i < t.shape[0]; i++) {
                    if (Math.abs(t.t[i][k]) > Math.abs(t.t[i_max][k])) {
                        i_max = i;
                    }
                }

                if (Math.abs(t.t[i_max][k]) == 0) {
                    k++
                } else {
                    if (h != i_max) {
                        [t.t[h], t.t[i_max]] = [t.t[i_max], t.t[h]]
                    }

                    for (let i = h + 1; i < t.shape[0]; i++) {
                        const f = t.t[i][k] / t.t[h][k]
                        t.t[i][k] = 0
                        for (let j = k + 1; j < t.shape[1]; j++) {
                            t.t[i][j] = t.t[i][j] - t.t[h][j] * f
                        }
                    }
                    h++
                    k++
                }
            }
        }
        return t
    }

    public swapRows(i, j) {
        [this.t[i], this.t[j]] = [this.t[j], this.t[i]]
    }

    public inv(): Tensor {
        if (this.dim != 2) {
            throw "inv(): Tensor is not a matrix."
        } else if (this.shape[0] != this.shape[1]) {
            throw "inv(): Tensor is not quadratic."
        } else {
            if (this.shape[0] == 1) {
                return new Tensor([[1 / this.t[0][0]]])
            } else if (this.shape[0] == 2) {
                return new Tensor([
                    [this.t[1][1], -this.t[0][1]],
                    [-this.t[1][0], this.t[0][0]]
                ]).mul(1 / ((this.t[0][0] * this.t[1][1]) - (this.t[0][1] * this.t[1][0])))
            } else if (this.shape[0] >= 3){
                const I = Tensor.createIdentityMatrix(this.shape[0])
                const t = this.copy(true)

                let lead = 0;
                for (let k = 0; k < this.shape[0]; k++) {
                    if (t.shape[1] <= lead) {
                        if (this.det() == 0) {
                            console.trace()
                            throw "inv(): Determinant of matrix is zero (0)!"
                        }
                        throw "inv(): Error!"
                    }

                    let i = k;
                    while (t.t[i][lead] == 0) {
                        i++;
                        if (t.shape[0] == i) {
                            i = k;
                            lead++;
                            if (t.shape[1] == lead) {
                                if (this.det() == 0) {
                                    console.trace()
                                    throw "inv(): Determinant of matrix is zero (0)!"
                                }
                            }
                        }
                    }

                    t.swapRows(i,k)
                    I.swapRows(i,k)

                    let val = t.t[k][lead]
                    for (let j = 0; j < t.shape[1]; j++) {
                        t.t[k][j] /= val
                        I.t[k][j] /= val
                    }

                    for (let i = 0; i < t.shape[0]; i++) {
                        if (i === k) continue;
                        val = t.t[i][lead];
                        for (let j = 0; j < t.shape[1]; j++) {
                            t.t[i][j] -= val * t.t[k][j];
                            I.t[i][j] -= val * I.t[k][j];
                        }
                    }
                    lead++;
                }
                return I
            }
        }
    }

    public extend(b: Tensor, axis=1) {
        if (axis == 1) {
            if (b.dim != 1 && b.shape[0] == this.shape[0]) {
                throw "extend(): Vector wrong number of elements!"
            }

            const t = new Tensor([this.shape[0], this.shape[1] + 1], true)
            t.iterate((i, j) => {
                if (j == this.shape[1]) {
                    t.t[i][j] = b.t[i]
                } else {
                    t.t[i][j] = this.t[i][j]
                }
            })
            return t
        }
    }

    public lu() {
        if (this.dim != 2) {
            throw "to_upper_triangular(): This must be a matrix!"
        }
        const U = this.copy(true)
        const L = this.copy()
        const P = Tensor.createIdentityMatrix(this.dim)

        for (let k = 0; k < this.shape[1]; k++) {
            let max_index = k
            for (let i = k + 1; i < this.shape[1]; i++) {
                if (Math.abs(U.t[i][k]) > Math.abs(U.t[max_index][k])) {
                    max_index = i
                }
            }

            U.swapRows(k,max_index)
            L.swapRows(k,max_index)
            P.swapRows(k,max_index)

            for(let i = k + 1; i < this.shape[1]; i++) {
                const factor = -U.t[i][k]/U.t[k][k]
                U.t[i][k] = 0
                for (let j = k + 1; j <= this.shape[1]; j++) {
                    U.t[i][j] = U.t[i][j] + factor * U.t[k][j]
                }
            }
        }

        return [L, U, P]

    }

    public solve(b: Tensor) {
        const tr = this.transpose()
        return tr.dot(this).inv().dot(tr.dot(b))
    }

    public solve2(b: Tensor) {
        const [m, n] = this.shape
        const aug = this.extend(b, 1)

        for (let k = 0; k < this.shape[1]; k++) {
            let max_index = k
            for (let i = k + 1; i < this.shape[1]; i++) {
                if (Math.abs(aug.t[i][k]) > Math.abs(aug.t[max_index][k])) {
                    max_index = i
                }
            }

            aug.swapRows(k,max_index)

            for(let i = k + 1; i < this.shape[1]; i++) {
                const factor = -aug.t[i][k]/aug.t[k][k]
                aug.t[i][k] = 0
                for (let j = k + 1; j <= this.shape[1]; j++) {
                    aug.t[i][j] = aug.t[i][j] + factor * aug.t[k][j]
                }
            }
        }

        const x = b.copy()
        for (let k = m - 1; k >= 0; k--) {
            x.t[k] = aug.t[k][n]
            for (let i = k + 1; i < n; i++) {
                x.t[k] = <number> x.t[k] - aug.t[k][i] * <number> x.t[i]
            }
            x.t[k] = <number> x.t[k] / aug.t[k][k]
        }

        return x
    }

    public det() {
        if (this.dim == 2) {
            if (this.shape[0] == this.shape[1]) {
                let sum = 1
                const row_reduced_matrix = this.rref()
                for (let i = 0; i < this.shape[0]; i++) {
                    sum *= row_reduced_matrix.t[i][i]
                }
                return sum
            } else {
                throw "det(): Tensor must be a square"
            }
        } else {
            throw "det(): Tensor must be an matrix."
        }
    }

    cond() {
        return this.inv().norm() * this.norm()
    }

    concatenate(t: Tensor, direction: "h" | "v" = "h"): Tensor {
        if (this.dim == 2 && t.dim == 2) {

            if (direction === "h" && this.dim[0] !== t.dim[0]) {
                console.error('Matrices must have the same number of rows for horizontal concatenation.');
                return null;
            }

            if (direction === "v") {
                return new Tensor([...this.t, ...(t.t as Float64Array[][]).map(row => [...row])]);
            } else {
                return new Tensor((this.t as Float64Array[][]).map((row, index) => [...row, ...(t.t as Float64Array[])[index]]));
            }
        } else {
            throw "concatenate(): t_1 or t_2 are not matrices"
        }

    }

}