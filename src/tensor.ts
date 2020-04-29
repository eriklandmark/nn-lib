import Vector from "./vector";
import Matrix from "./matrix";

export default class Tensor {

    tensor: Float32Array[][] = [];

    public get: Function = (i: number, j: number, k: number) => {
        if(isNaN(this.tensor[i][j][k])) {
            console.trace()
            throw "Getting an NaN..."
        }
        return this.tensor[i][j][k]
    };
    public set: Function = (i: number, j: number, k: number, n: number) => {
        if (isNaN(n)) {
            console.trace()
            throw "Number is NaN..."
        }
        this.tensor[i][j][k] = n;
    };
    public count: Function = () => {
        return this.dim().c * this.dim().r * this.dim().d;
    };

    public dim() {
        return {
            r: this.tensor.length,
            c: this.tensor[0] ? this.tensor[0].length : 0,
            d: this.tensor[0][0] ? this.tensor[0][0].length : 0
        }
    }

    public shape(): number[] {
        return [this.dim().r, this.dim().c, this.dim().d]
    }

    constructor(v: number[][][] | Float32Array[][] = []) {
        if (v.length > 0 && v[0][0] instanceof Float32Array) {
            this.tensor = <Float32Array[][]> v
        } else {
            for (let i = 0; i < v.length; i++) {
                this.tensor.push([])
                for (let j = 0; j < v[i].length; j++) {
                    this.tensor[i].push(Float32Array.from(<Array<number>>v[i][j]))
                }
            }
        }
    }

    public createEmptyArray(rows: number, columns: number, depth: number) {
        this.tensor = []
        for (let i = 0; i < rows; i++) {
            this.tensor.push([]);
            for (let j = 0; j < columns; j++) {
                this.tensor[i].push(new Float32Array(depth).fill(0))
            }
        }
    }

    public static fromJsonObject(obj: any[][]) {
        return new Tensor(obj.map((row: any[]) => {
            return row.map((col: any) => {
                return Object.keys(col).map((item, index) => col[index.toString()])
            })
        }))
    }

    public toNumberArray(): number[][] {
        return this.tensor.map((array) => array.map((floatArray) => [].slice.call(floatArray)))
    }

    public iterate(func: Function, channel_first = false): void {
        if (channel_first) {
            for (let k: number = 0; k < this.dim().d; k++) {
                for (let i: number = 0; i < this.dim().r; i++) {
                    for (let j: number = 0; j < this.dim().c; j++) {
                        func(i, j, k);
                    }
                }
            }
        } else {
            for (let i: number = 0; i < this.dim().r; i++) {
                for (let j: number = 0; j < this.dim().c; j++) {
                    for (let k: number = 0; k < this.dim().d; k++) {
                        func(i, j, k);
                    }
                }
            }
        }

    }

    public toString = (max_rows: number = 10): string => {
        if (this.tensor.length == 0) {
            return "Tensor: 0x0x0 []"
        } else {
            let maxCharCount = 0;
            this.iterate((i: number, j: number, k: number) => {
                let val = this.get(i, j, k).toString()
                if (val.length > maxCharCount) maxCharCount = val.length
            })
            maxCharCount = Math.min(maxCharCount, 7)
            let string = `Tensor: ${this.dim().r}x${this.dim().c}x${this.dim().d} [\n`
            for (let d = 0; d < this.dim().d; d++) {
                string += this.tensor.slice(0, Math.min(max_rows, this.tensor.length)).reduce((acc, i) => {
                    acc += i.slice(0, Math.min(max_rows, i.length)).reduce((s, j) => {
                        s += " ".repeat(Math.max(maxCharCount - j[d].toString().length, 0))
                        s += j[d].toString().substr(0, maxCharCount) + " ";
                        return s;
                    }, "    ")
                    acc += i.length > max_rows ? " ... +" + (i.length - max_rows) + " elements\n" : "\n"
                    return acc;
                }, "") + (this.tensor.length > max_rows ?
                    "    ... +" + (this.tensor.length - max_rows) + " rows \n" : "\n")
            }
            return string + "]"
        }
    }

    public copy(full: boolean = true) {
        let t = new Tensor()
        t.createEmptyArray(this.dim().r, this.dim().c, this.dim().d)
        if (full) {
            t.iterate((i: number, j: number, k: number) => {
                t.set(i, j, k, this.get(i, j, k))
            })
        }
        return t
    }

    public populateRandom() {
        this.iterate((i: number, j: number, k: number) => {
            this.set(i, j, k, Math.random() * 2 - 1)
        })
    }

    public empty(): boolean {
        return this.dim().c == 0 || this.dim().r == 0 || this.dim().d == 0
    }

    public vectorize(channel_first = false): Vector {
        const v = new Vector(this.count())
        let index = 0;
        this.iterate((i: number, j: number, k: number) => {
            v.set(index, this.get(i, j, k))
            index += 1
        }, channel_first)
        return v
    }

    public div(val: number | Tensor): Tensor {
        let t = this.copy(false);
        if (val instanceof Tensor) {
            if (t.dim().r != this.dim().r || t.dim().c != this.dim().c || t.dim().d != this.dim().d) {
                console.trace()
                throw "Tensor Division: Not the same dimension"
            }
            this.iterate((i: number, j: number, k: number) => {
                t.set(i, j, k, this.get(i, j, k) / val.get(i, j, k))
            });
        } else {
            this.iterate((i: number, j: number, k: number) => {
                t.set(i, j, k, this.get(i, j, k) / val)
            });
        }
        return t
    }

    public mul(val: number | Tensor): Tensor {
        let t = this.copy(false);
        if (val instanceof Tensor) {
            if (t.dim().r != this.dim().r || t.dim().c != this.dim().c || t.dim().d != this.dim().d) {
                console.trace()
                throw "Tensor Multiplication: Not the same dimension"
            }
            this.iterate((i: number, j: number, k: number) => {
                t.set(i, j, k, this.get(i, j, k) * val.get(i, j, k))
            });
        } else {
            this.iterate((i: number, j: number, k: number) => {
                t.set(i, j, k, this.get(i, j, k) * val)
            });
        }
        return t
    }

    public sub(val: number | Tensor): Tensor {
        let t = this.copy(false);
        if (val instanceof Tensor) {
            if (t.dim().r != this.dim().r || t.dim().c != this.dim().c || t.dim().d != this.dim().d) {
                console.trace()
                throw "Tensor Subtraction: Not the same dimension"
            }
            this.iterate((i: number, j: number, k: number) => {
                t.set(i, j, k, this.get(i, j, k) - val.get(i, j, k))
            });
        } else {
            this.iterate((i: number, j: number, k: number) => {
                t.set(i, j, k, this.get(i, j, k) - val)
            });
        }
        return t
    }

    public add(val: number | Tensor): Tensor {
        let t = this.copy(false);
        if (val instanceof Tensor) {
            if (t.dim().r != this.dim().r || t.dim().c != this.dim().c || t.dim().d != this.dim().d) {
                console.trace()
                throw "Tensor Subtraction: Not the same dimension"
            }
            this.iterate((i: number, j: number, k: number) => {
                t.set(i, j, k, this.get(i, j, k) + val.get(i, j, k))
            });
        } else {
            this.iterate((i: number, j: number, k: number) => {
                t.set(i, j, k, this.get(i, j, k) + val)
            });
        }
        return t
    }

    padding(padding_height: number, padding_width: number) {
        const t = new Tensor()
        t.createEmptyArray(
            2 * padding_height + this.dim().r, 2 * padding_width + this.dim().c, this.dim().d
        )

        for (let i = 0; i < this.dim().r; i++) {
            for (let j = 0; j < this.dim().c; j++) {
                for (let c = 0; c < this.dim().d; c++) {
                    t.set(i + padding_height, j + padding_width, c, this.get(i,j,c))
                }
            }
        }
        return t
    }

    im2patches(patch_height: number, patch_width: number, filter_height: number, filter_width: number): Matrix {
        const cols = []
        for (let r = 0; r < patch_height; r++) {
            for (let c = 0; c < patch_width; c++) {
                const v = []
                for (let c_f_c = 0; c_f_c < this.dim().d; c_f_c++) {
                    for (let c_f_h = 0; c_f_h < filter_height; c_f_h++) {
                        for (let c_f_w = 0; c_f_w < filter_width; c_f_w++) {
                            v.push(this.get(r + c_f_h, c + c_f_w, c_f_c))
                        }
                    }
                }
                cols.push(new Vector(v))
            }
        }

        return new Matrix(cols)
    }

    rotate180() {
        const t = this.copy(false)
        this.iterate((i: number, j: number, k: number) => {
            t.set(this.dim().r - 1 - i, this.dim().c - 1 - j , k, this.get(i,j,k))
        })
        return t
    }

    pow(number: number): Tensor {
        let t = this.copy(false);
        this.iterate((i: number, j: number, k: number) => {
            t.set(i, j, k, this.get(i, j, k) ** number)
        });
        return t
    }

    sqrt(): Tensor {
        let t = this.copy(false);
        this.iterate((i: number, j: number, k: number) => {
            t.set(i, j, k, Math.sqrt(this.get(i, j, k)))
        });
        return t
    }
}