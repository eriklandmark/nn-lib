export default class Vector {
    vector: Float64Array;

    public size: Function = (): number => {
        return this.vector.length
    };
    public get: Function = (i: number): number => {
        return this.vector[i]
    };
    public set: Function = (i: number, n: number): void => {
        this.vector[i] = n
    };

    constructor(defaultValue: Float64Array | number[] | number = new Float64Array(0)) {
        if (defaultValue instanceof Float64Array) {
            this.vector = defaultValue;
        } else if (typeof defaultValue == "number") {
            this.vector = new Float64Array(defaultValue);
        } else {
            this.vector = Float64Array.from(defaultValue);
        }
    }

    public static fromBuffer(buff: Buffer): Vector {
        let v: Vector = new Vector(buff.length)
        for (let i = 0; i < v.size(); i++) {
            v.set(i, buff[i]);
        }
        return v
    }

    public static toCategorical(index: number, size: number) {
        const v = new Vector(new Float64Array(size).fill(0));
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
}