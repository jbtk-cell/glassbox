/** Every way of zooming and panning a canvas view, shared by the map and the Neurons view:
 *  wheel and trackpad scroll zoom at the pointer, pinch zooms (and never zooms the page),
 *  touch drags and pinches, + / - / 0 keys, and double-click to zoom in.
 *  Mouse dragging stays in each view, because it shares mouse-down with clicking a cell. */
import { useCallback, useEffect, useRef, type Dispatch, type RefObject, type SetStateAction } from 'react';
import { zoomAt, pan, pinchMove, wheelFactor, ZOOM_MIN, ZOOM_MAX, type Camera } from './camera';

/** How much one press of a zoom button or key, or one double-click, changes the zoom. */
export const BUTTON_STEP = 1.6;
export const DOUBLE_CLICK_STEP = 2;

type SetCam = Dispatch<SetStateAction<Camera>>;
/** Safari reports a trackpad pinch as gesture events carrying a cumulative scale. */
type GestureEvent = Event & { scale: number; clientX: number; clientY: number };

export function usePanZoom(wrapRef: RefObject<HTMLElement | null>, setCam: SetCam, fitAll: () => void, maxZoom = ZOOM_MAX) {
  const fitRef = useRef(fitAll); fitRef.current = fitAll;

  const zoomBy = useCallback((factor: number) => {
    const el = wrapRef.current; if (!el) return;
    setCam(c => zoomAt(c, el.clientWidth / 2, el.clientHeight / 2, factor, ZOOM_MIN, maxZoom));
  }, [wrapRef, setCam, maxZoom]);

  useEffect(() => {
    const el = wrapRef.current; if (!el) return;
    const local = (x: number, y: number): [number, number] => { const b = el.getBoundingClientRect(); return [x - b.left, y - b.top]; };

    // React registers wheel listeners as passive, which cannot stop the page from scrolling or pinch-zooming.
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const [sx, sy] = local(e.clientX, e.clientY);
      const factor = wheelFactor(e);
      setCam(c => zoomAt(c, sx, sy, factor, ZOOM_MIN, maxZoom));
    };

    let lastScale = 1;
    const onGestureStart = (e: Event) => { e.preventDefault(); lastScale = 1; };
    const onGestureChange = (e: Event) => {
      e.preventDefault();
      const g = e as GestureEvent; const factor = g.scale / lastScale; lastScale = g.scale;
      const [sx, sy] = local(g.clientX, g.clientY);
      setCam(c => zoomAt(c, sx, sy, factor, ZOOM_MIN, maxZoom));
    };

    // Touch: one finger drags, two fingers pinch. Mouse and pen are left to the view's own handlers.
    const touches = new Map<number, [number, number]>();
    const onPointerDown = (e: PointerEvent) => { if (e.pointerType === 'touch') touches.set(e.pointerId, local(e.clientX, e.clientY)); };
    const onPointerMove = (e: PointerEvent) => {
      if (e.pointerType !== 'touch' || !touches.has(e.pointerId)) return;
      const before = touches.get(e.pointerId)!; const now = local(e.clientX, e.clientY);
      if (touches.size === 1) setCam(c => pan(c, now[0] - before[0], now[1] - before[1]));
      else if (touches.size === 2) {
        const other = [...touches.entries()].find(([id]) => id !== e.pointerId)![1];
        setCam(c => pinchMove(c, before, other, now, other, ZOOM_MIN, maxZoom));
      }
      touches.set(e.pointerId, now);
    };
    const onPointerEnd = (e: PointerEvent) => { touches.delete(e.pointerId); };

    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;   // leave the browser's own zoom shortcuts alone
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
      if (e.key === '+' || e.key === '=') zoomBy(BUTTON_STEP);
      else if (e.key === '-' || e.key === '_') zoomBy(1 / BUTTON_STEP);
      else if (e.key === '0') fitRef.current();
      else return;
      e.preventDefault();
    };

    el.addEventListener('wheel', onWheel, { passive: false });
    el.addEventListener('gesturestart', onGestureStart);
    el.addEventListener('gesturechange', onGestureChange);
    el.addEventListener('pointerdown', onPointerDown);
    el.addEventListener('pointermove', onPointerMove);
    el.addEventListener('pointerup', onPointerEnd);
    el.addEventListener('pointercancel', onPointerEnd);
    window.addEventListener('keydown', onKey);
    return () => {
      el.removeEventListener('wheel', onWheel);
      el.removeEventListener('gesturestart', onGestureStart);
      el.removeEventListener('gesturechange', onGestureChange);
      el.removeEventListener('pointerdown', onPointerDown);
      el.removeEventListener('pointermove', onPointerMove);
      el.removeEventListener('pointerup', onPointerEnd);
      el.removeEventListener('pointercancel', onPointerEnd);
      window.removeEventListener('keydown', onKey);
    };
  }, [wrapRef, setCam, zoomBy, maxZoom]);

  /** Double-click zooms in on the spot, as in a maps app; with Shift it zooms back out. */
  const onDoubleClick = useCallback((e: { clientX: number; clientY: number; shiftKey: boolean; target: EventTarget }) => {
    const el = wrapRef.current; if (!el || (e.target as HTMLElement).closest('.map-zoom')) return;
    const b = el.getBoundingClientRect();
    setCam(c => zoomAt(c, e.clientX - b.left, e.clientY - b.top, e.shiftKey ? 1 / DOUBLE_CLICK_STEP : DOUBLE_CLICK_STEP, ZOOM_MIN, maxZoom));
  }, [wrapRef, setCam, maxZoom]);

  return { zoomBy, onDoubleClick };
}
