import { useEffect } from 'react';
import { Shell } from './ui/Shell';
import { useStore } from './app/store';

export function App() {
  useEffect(() => {
    if (import.meta.env.DEV) (window as unknown as { glassbox: typeof useStore }).glassbox = useStore;
    const s = useStore.getState();
    if (!s.corpus) { s.applyText(); s.setPrompt('you coming'); }
  }, []);
  return <Shell />;
}
