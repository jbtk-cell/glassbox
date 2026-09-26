/** A small colour legend drawn in a canvas corner, shared by the Network and Neurons views. */
import { diverging, sequential, gradientMap } from './colormap';

const rgb = (c: [number, number, number]) => `rgb(${c[0] | 0}, ${c[1] | 0}, ${c[2] | 0})`;

export type LegendPhase = 'forward' | 'backward' | 'update';

/** Draws the legend with its bottom-right corner at (right, bottom). Returns the box drawn. */
export function drawLegend(g: CanvasRenderingContext2D, right: number, bottom: number, phase: LegendPhase): { x: number; y: number; w: number; h: number } {
  const rows: { swatch: (x: number, y: number) => void; text: string }[] = [];
  const bar = (fn: (t: number) => [number, number, number]) => (x: number, y: number) => {
    for (let i = 0; i < 36; i++) { g.fillStyle = rgb(fn(i / 35)); g.fillRect(x + i, y, 1.5, 10); }
    g.strokeStyle = '#999'; g.lineWidth = 0.5; g.strokeRect(x, y, 36, 10);
  };
  const square = (fill: string, stroke: string, lw: number) => (x: number, y: number) => { g.fillStyle = fill; g.fillRect(x + 12, y, 12, 10); g.strokeStyle = stroke; g.lineWidth = lw; g.strokeRect(x + 12, y, 12, 10); };
  if (phase === 'forward') {
    rows.push({ swatch: bar(t => diverging(t * 2 - 1, 1)), text: 'negative  0  positive' });
    rows.push({ swatch: bar(t => sequential(t)), text: '0 to 1: attention, probabilities' });
  } else {
    rows.push({ swatch: bar(t => gradientMap(t * 2 - 1, 1)), text: phase === 'backward' ? 'gradient: raising this number lowers the loss  /  raises it' : 'update: moved down  /  moved up' });
  }
  rows.push({ swatch: square('#f5efe6', '#e08a1e', 2), text: 'learned numbers (training changes them)' });
  rows.push({ swatch: square('#f4f4f4', '#222', 1), text: 'numbers computed from the words' });

  g.font = '10.5px system-ui';
  const w = 8 + 36 + 8 + Math.max(...rows.map(r => g.measureText(r.text).width)) + 10, rowH = 16, h = rows.length * rowH + 10;
  const x = right - w, y = bottom - h;
  g.fillStyle = 'rgba(255,255,255,0.94)'; g.strokeStyle = '#cfcfcf'; g.lineWidth = 1;
  g.beginPath(); g.roundRect(x, y, w, h, 4); g.fill(); g.stroke();
  rows.forEach((r, i) => {
    const ry = y + 6 + i * rowH;
    r.swatch(x + 8, ry);
    g.fillStyle = '#333'; g.textAlign = 'left'; g.textBaseline = 'middle'; g.font = '10.5px system-ui';
    g.fillText(r.text, x + 8 + 36 + 8, ry + 5);
  });
  return { x, y, w, h };
}

/** A one-line banner with a white backing, top-left, used for the phase notice. */
export function drawBanner(g: CanvasRenderingContext2D, x: number, y: number, text: string, colour = '#7c3d00'): void {
  g.font = '600 11.5px system-ui'; g.textAlign = 'left'; g.textBaseline = 'middle';
  const w = g.measureText(text).width + 16;
  g.fillStyle = 'rgba(255,248,235,0.95)'; g.strokeStyle = '#e0b060'; g.lineWidth = 1;
  g.beginPath(); g.roundRect(x, y, w, 22, 4); g.fill(); g.stroke();
  g.fillStyle = colour; g.fillText(text, x + 8, y + 11);
}
