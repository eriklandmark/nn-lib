import Matrix from "./matrix";

export default class Vector {
    vector: Float64Array;

    public size : Function = ():number => {return this.vector.length};
    public get: Function = (i:number):number => {return this.vector[i]};
    public set: Function = (i:number, n:number):void => {this.vector[i] = n};

    constructor(defaultValue: Float64Array | number[] | number = new Float64Array(0)) {
        if (defaultValue instanceof Float64Array) {
            this.vector = defaultValue;
        } else if (typeof defaultValue == "number") {
            this.vector = new Float64Array(defaultValue);
        } else {
            this.vector = Float64Array.from(defaultValue);
        }
    }

    public static toCategorical(index: number, size: number) {
        const v = new Vector(new Float64Array(size).fill(0));
        v.set(index, 1);
        return v
    }

    public toString = (vertical: boolean = true) : string => {
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
        this.iterate((_, index) => {
            this.set(index, (Math.random() - 0.5) * 0.1)
        })
    }

    public iterate(func: Function): void {
        this.vector.forEach((value, index) => {func(value, index)})
    }

    public add(b: number | Vector): Vector {
        let v = new Vector(this.vector);
        if (typeof b == "number") {
            let scalar: number = <number> b;
            this.iterate((val, i) => {v.set(i, val + scalar)});
            return v
        } else if (b instanceof Vector) {
            if (b.size() != this.size()) throw "Vectors to add aren't the same size..";
            this.iterate((val, i) => {v.set(i, val + b.get(i))});
            return v
        }
    }

    public sub(b: number | Vector): Vector {
        let v = new Vector(this.vector);
        if (typeof b == "number") {
            let scalar: number = <number> b;
            this.iterate((val, i) => {v.set(i, val - scalar)});
            return v
        } else if (b instanceof Vector) {
            if (b.size() != this.size()) throw "Vectors to subtract aren't the same size..";
            this.iterate((val, i) => {v.set(i, val - b.get(i))});
            return v
        }
    }

    public mul(input: number | Vector): Vector {
        let v = new Vector(this.vector);
        if (input instanceof Vector) {
            if (input.size() != this.size()) throw "Vectors to multiply aren't the same size..";
            this.iterate((val, i) => {v.set(i, val * input.get(i))});
        } else {
            this.iterate((val, i) => {v.set(i, val * input)});
        }
        return v
    }

    public div(scalar: number): Vector {
        let v = new Vector(this.vector);
        this.iterate((val, i) => {v.set(i, val / scalar)});
        return v
    }

    public mean(): number {
        const sum = this.vector.reduce((acc, val) => acc + val)
        return sum / this.size();
    }
}