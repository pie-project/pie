import { useState, ReactNode } from 'react';
import styles from './FairNotes.module.css';

interface NoteSection {
  title: string;
  content: ReactNode;
}

const NOTES: NoteSection[] = [
  {
    title: 'Where standard prefix caching is equally good',
    content: (
      <ul>
        <li>
          <strong>Same-session multi-turn:</strong> append-only conversation
          history hits the prefix cache perfectly. Both systems reuse 100% of
          prior KV.
        </li>
        <li>
          <strong>Same-channel cross-session:</strong> when the system prompt
          does not change between requests, standard prefix caching reuses the
          full prefix just like Pie.
        </li>
        <li>
          <strong>Append-only tool results:</strong> tool call outputs appended
          after the system prompt extend the prefix without invalidation.
          Standard caching handles this natively.
        </li>
      </ul>
    ),
  },
  {
    title: 'What we do NOT claim',
    content: (
      <ul>
        <li>
          <strong>Decode speed:</strong> Pie uses the same GPU decode path.
          Token generation throughput is identical once prefill completes.
        </li>
        <li>
          <strong>Steady-state advantage:</strong> for long-running single-session
          workloads that never switch context, there is no measurable difference.
        </li>
        <li>
          <strong>Free lunch:</strong> Pie inferlets add a thin coordination
          layer. The overhead is small (~0.6ms per IPC round-trip) but nonzero.
        </li>
      </ul>
    ),
  },
  {
    title: 'Alternatives to Pie',
    content: (
      <ul>
        <li>
          <strong>Prompt reordering:</strong> an application could reorder its
          own prompt to put dynamic sections at the end. Pie's inferlet does
          this transparently without upstream changes.
        </li>
        <li>
          <strong>CPU KV offloading:</strong> some inference systems support
          offloading evicted KV blocks to CPU DRAM (~2s reload). Under no
          memory pressure, both approaches retain KV equally.
        </li>
        <li>
          <strong>Warm-up requests:</strong> sending a dummy request on cold
          start populates the prefix cache. Pie's checkpoint approach works
          across engine restarts without extra orchestration.
        </li>
      </ul>
    ),
  },
];

function CollapsibleNote({ title, content }: NoteSection) {
  const [open, setOpen] = useState(false);
  const cn = (...classes: (string | false | undefined)[]) =>
    classes.filter(Boolean).join(' ');

  return (
    <div className={styles.card}>
      <button
        className={styles.cardHeader}
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        <span className={cn(styles.chevron, open && styles.chevronOpen)}>
          &#9654;
        </span>
        <h4 className={styles.cardTitle}>{title}</h4>
      </button>
      {open && <div className={styles.cardBody}>{content}</div>}
    </div>
  );
}

export default function FairNotes() {
  return (
    <div className={styles.wrapper}>
      <h3 className={styles.title}>Honest Assessment</h3>
      <div className={styles.grid}>
        {NOTES.map((note) => (
          <CollapsibleNote key={note.title} {...note} />
        ))}
      </div>
    </div>
  );
}
