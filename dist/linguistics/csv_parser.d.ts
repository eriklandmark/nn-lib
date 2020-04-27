export default class CsvParser {
    static parse(data: string, isPath?: boolean): (string | number)[][];
    static filterColumns(data: (number | string)[][], columns: number[]): (string | number)[][];
}
