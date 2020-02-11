export default class Vector {
    vector: Float64Array;

    public size : Function = ():number => {return this.vector.length};
    public get: Function = (i:number):number => {return this.vector[i]};
    public set: Function = (i:number, n:number):void => {this.vector[i] = n};

    constructor(defaultValue: Float64Array = new Float64Array(0)) {
        this.vector = defaultValue;
    }

    public toString = () : string => {
        return this.vector.reduce((acc: any, v) => {
            acc += v.toString() + " "
            return acc;
        }, "[ ") + "]"
    }

}