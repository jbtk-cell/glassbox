import { useEffect } from 'react';
import { FlowView } from './views/flow/FlowView';
import { useStore } from './app/store';
import { PRESETS } from './presets';

// Temporary harness for the Flow view; replaced by the Shell in the next task.
export function App() {
  useEffect(() => {
    (window as unknown as { glassbox: typeof useStore }).glassbox = useStore;
    const s = useStore.getState();
    s.setText(PRESETS[1].text, PRESETS[1].id); s.applyText();
    s.setPrompt('twinkle twinkle little'); s.recordPrompt();
    const onKey = (e: KeyboardEvent) => {
      const st = useStore.getState();
      if (e.key === 'f' || e.key === 'ArrowRight') st.next();
      else if (e.key === 'b' || e.key === 'ArrowLeft') st.prev();
      else if (e.key === 't') st.recordTrainingTrace();
      else if (e.key === 'Home') st.toStart(); else if (e.key === 'End') st.toEnd();
    };
    window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey);
  }, []);
  return <div style={{ position: 'fixed', inset: 0 }}><FlowView /></div>;
}
