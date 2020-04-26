import fs from "fs"
export default class CsvParser {

    public static parse(data: string, isPath = false) {
        let content: string
        if(isPath) {
            content = fs.readFileSync(data, {encoding: "utf-8"})
        } else {
            content = data
        }

        const lines = content.split("\n")

        return lines.map((line) => line.trim().split("\t").map((cell) => {
            const n = parseFloat(cell.trim())
            return isFinite(n)? n: cell.trim()
        }))
    }

    public static filterColumns(data: (number | string)[][], columns: number[]) {
        return data.map((line) => {
            return line.filter((_, index) => columns.includes(index))
        })
    }

}