import groupchat from './groupchat.txt?raw';
import nursery from './nursery.txt?raw';
import alice from './alice.txt?raw';

export interface Preset { id: string; name: string; blurb: string; text: string }

export const PRESETS: Preset[] = [
  { id: 'groupchat', name: 'Group chat', blurb: 'Short text-message exchanges. Trains in seconds and sounds like someone you know.', text: groupchat },
  { id: 'nursery', name: 'Nursery rhymes', blurb: 'Public-domain rhymes. Very repetitive, so the model memorises them almost instantly.', text: nursery },
  { id: 'alice', name: 'Alice', blurb: 'The opening of Alice in Wonderland (1865). Bigger vocabulary; watch it struggle to generalise.', text: alice },
];
