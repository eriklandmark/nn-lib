import Vector from "./vector";
import {KernelFunction} from "gpu.js";

export default class Matrix {
    matrix: Array<Float32Array>;
    get: Function;
    set: Function;
    count: Function;
    constructor(defaultValue?: Array<Array<number>> | Array<Float32Array> | Array<Vector>);
    createEmptyArray(rows: number, columns: number): void;
    dim(): {
        r: number;
        c: number;
    };
    toString: (max_rows?: number) => string;
    static fromJsonObject(obj: any[]): Matrix;
    toNumberArray(): number[][];
    copy(): Matrix;
    iterate(func: Function): void;
    where(scalar: number): number[];
    populateRandom(): void;
    empty(): boolean;
    static addGpu(): KernelFunction;
    static subGpu(): KernelFunction;
    static multiplyGpu(): KernelFunction;
    static mmGpu(): KernelFunction;
    mm(b: Matrix | Vector, gpu?: boolean): Matrix | Vector;
    mmAsync(b: Matrix | Vector): Promise<Matrix | Vector>;
    add(b: number | Matrix): Matrix;
    sub(b: number | Matrix): Matrix;
    mul(b: number | Matrix): Matrix;
    pow(scalar: number): Matrix;
    exp(): Matrix;
    log(): Matrix;
    sum(axis?: number, keepDims?: boolean): number | Matrix;
    div(scalar: number | Matrix): Matrix;
    transpose(): Matrix;
    argmax(i?: number, row?: boolean): number;
    inv(): Matrix;
}
