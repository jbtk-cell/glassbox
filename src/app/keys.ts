import { useEffect } from 'react';
import { useStore } from './store';

/** Global keys, ignored while typing in a form control. F/B step operations; T records a training step; Space toggles generation. */
export function useKeys(togglePlay: () => void) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
      const s = useStore.getState();
      switch (e.key) {
        case 'f': case 'F': case 'ArrowRight': s.next(); break;
        case 'b': case 'B': case 'ArrowLeft': s.prev(); break;
        case 'Home': s.toStart(); break;
        case 'End': s.toEnd(); break;
        case ' ': togglePlay(); break;
        case 't': case 'T': s.recordTrainingTrace(); break;
        default: return;
      }
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [togglePlay]);
}
