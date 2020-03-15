import Vector from "./vector";

export default class Tensor {

    tensor: Float32Array[][] = [];

    public get: Function = (i: number, j: number, k: number) => {
        return this.tensor[i][j][k]
    };
    public set: Function = (i: number, j: number, k: number, n: number) => {
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

    public iterate(func: Function): void {
        for (let i: number = 0; i < this.dim().r; i++) {
            for (let j: number = 0; j < this.dim().c; j++) {
                for (let k: number = 0; k < this.dim().d; k++) {
                    func(i, j, k);
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

    public vectorize(): Vector {
        const v = new Vector(this.count())
        let index = 0;
        this.iterate((i: number, j: number, k: number) => {
            v.set(index, this.get(i, j, k))
            index += 1
        })
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
}