export interface Tensor { name: string; shape: number[]; data: Float64Array }

export function size(shape: number[]): number { return shape.reduce((a, b) => a * b, 1); }

export function tensor(name: string, shape: number[], data?: Float64Array): Tensor {
  const n = size(shape);
  if (data && data.length !== n) throw new Error(`tensor ${name}: data length ${data.length} != shape size ${n}`);
  return { name, shape: [...shape], data: data ?? new Float64Array(n) };
}

export function zerosLike(t: Tensor, name: string): Tensor { return tensor(name, t.shape); }
