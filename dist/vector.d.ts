/// <reference types="node" />
import Tensor from "./tensor";
export default class Vector {
    vector: Float64Array;
    size: Function;
    get: Function;
    set: Function;
    constructor(defaultValue?: Float64Array | number[] | number);
    static fromJsonObj(obj: any): Vector;
    static fromBuffer(buff: Buffer): Vector;
    static toCategorical(index: number, size: number): Vector;
    toString: (vertical?: boolean) => string;
    toNumberArray(): any;
    populateRandom(): void;
    iterate(func: Function): void;
    add(b: number | Vector): Vector;
    sub(b: number | Vector): Vector;
    mul(input: number | Vector): Vector;
    div(scalar: number): Vector;
    pow(scalar: number): Vector;
    exp(): Vector;
    sum(): number;
    mean(): number;
    argmax(): number;
    reshape(shape: number[]): Tensor;
}
