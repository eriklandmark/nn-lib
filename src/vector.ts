import Tensor from "./tensor";

export default class Vector {
    vector: Float32Array;

    public size: Function = (): number => {
        return this.vector.length
    };
    public get: Function = (i: number): number => {
        return this.vector[i]
    };
    public set: Function = (i: number, n: number): void => {
        this.vector[i] = n
    };

    constructor(defaultValue: Float32Array | number[] | number = new Float32Array(0)) {
        if (defaultValue instanceof Float32Array) {
            this.vector = defaultValue;
        } else if (typeof defaultValue == "number") {
            this.vector = new Float32Array(defaultValue);
        } else {
            this.vector = Float32Array.from(defaultValue);
        }
    }

    public static fromJsonObj(obj: any) {
        return new Vector(Object.keys(obj).map(
            (item, index) => {
                return obj[index.toString()]
            }
        ))
    }

    public static fromBuffer(buff: Buffer): Vector {
        let v: Vector = new Vector(buff.length)
        for (let i = 0; i < v.size(); i++) {
            v.set(i, buff[i]);
        }
        return v
    }

    public static toCategorical(index: number, size: number) {
        const v = new Vector(new Float32Array(size).fill(0));
        v.set(index, 1);
        return v
    }

    public toString = (vertical: boolean = true): string => {
        if (this.vector.length == 0) {
            return "Vector: []"
        } else {
            if (vertical) {
                return this.vector.reduce((acc, i) => {
                    acc += `    ${i}\n`
                    return acc;
                }, `Vector: [\n`) + " ]"
            } else {
                return this.vector.reduce((acc: any, v) => {
                    acc += v.toString() + " "
                    return acc;
                }, "Vector: [ ") + "]"
            }
        }
    }

    public copy(full = true): Vector {
        const v = new Vector(this.size())
        if(full) {
            v.vector = this.vector.copyWithin(0, this.size())
        }
        return v
    }

    public toNumberArray() {
        return [].slice.call(this.vector)
    }

    public populateRandom() {
        this.iterate((_: number, index: number) => {
            this.set(index, Math.random() * 2 - 1)
        })
    }

    public iterate(func: Function): void {
        this.vector.forEach((value, index) => {
            func(value, index)
        })
    }

    public add(b: number | Vector): Vector {
        let v = new Vector(this.size());
        if (b instanceof Vector) {
            if (b.size() != this.size()) throw "Vectors to add aren't the same size..";
            this.iterate((val: number, i: number) => {
                v.set(i, val + b.get(i))
            });
            return v
        } else {
            let scalar: number = <number>b;
            this.iterate((val: number, i: number) => {
                v.set(i, val + scalar)
            });
            return v
        }
    }

    public sub(b: number | Vector): Vector {
        let v = new Vector(this.size());
        if (b instanceof Vector) {
            if (b.size() != this.size()) throw "Vectors to subtract aren't the same size..";
            this.iterate((val: number, i: number) => {
                v.set(i, val - b.get(i))
            });
            return v
        } else {
            let scalar: number = <number>b;
            this.iterate((val: number, i: number) => {
                v.set(i, val - scalar)
            });
            return v
        }
    }

    public mul(input: number | Vector): Vector {
        let v = new Vector(this.size());
        if (input instanceof Vector) {
            if (input.size() != this.size()) {
                console.trace();
                throw "Vectors to multiply aren't the same size..";
            }
            this.iterate((val: number, i: number) => {
                v.set(i, val * input.get(i))
            });
        } else {
            this.iterate((val: number, i: number) => {
                v.set(i, val * input)
            });
        }
        return v
    }

    public div(scalar: number): Vector {
        let v = new Vector(this.size());
        this.iterate((val: number, i: number) => {
            v.set(i, val / scalar)
        });
        return v
    }

    public pow(scalar: number): Vector {
        let v = new Vector(this.size());
        this.iterate((val: number, i: number) => {
            v.set(i, val ** scalar)
        });
        return v
    }

    public exp(): Vector {
        let v = new Vector(this.size());
        this.iterate((val: number, i: number) => {
            v.set(i, Math.exp(val))
        });
        return v
    }

    public sum(): number {
        return this.vector.reduce((acc, val) => acc + val);
    }

    public mean(): number {
        return this.sum() / this.size();
    }

    public argmax() {
        return this.vector.reduce((acc: number, va: number, ind) => va > this.get(acc) ? ind : acc, 0)
    }

    public reshape(shape: number[]) {
        if(this.size() != shape.reduce((acc, n) => acc * n, 1)) {
            throw "Product of shape must be the same as size of vector!"
        }
        const t = new Tensor();
        t.createEmptyArray(shape[0], shape[1], shape[2])
        let [h, w, d] = shape

        this.iterate((val: number, i: number) => {
            const r = Math.floor(i / (w*d))
            const c = Math.floor(i / (d) - (r*w))
            const g = Math.floor(i - (c*d) - (r*w*d))
            t.set(r,c,g, val)
        })
        return t
    }

    public normalize() {
        return this.div(this.size())
    }
}