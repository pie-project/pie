import { useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import useBaseUrl from '@docusaurus/useBaseUrl';
import DemoPlayer from './DemoPlayer';
import type { DemoTrace } from './types';
import styles from './index.module.css';

export type DemoId =
  | 'self-correct'
  | 'custom-spec'
  | 'parallel-fork'
  | 'grammar'
  | 'persistent-kv'
  | 'filesystem'
  | 'watermark';

export interface DemoMeta {
  id: DemoId;
  tab: string;
  metric: string;
}

interface Props {
  demos: DemoMeta[];
}

function useTrace(id: string) {
  const url = useBaseUrl(`/demos/${id}.json`);
  const [trace, setTrace] = useState<DemoTrace | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setTrace(null);
    setError(null);
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`status ${r.status}`);
        return r.json();
      })
      .then((data: DemoTrace) => {
        if (!cancelled) setTrace(data);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [url]);

  return { trace, error };
}

export default function HomeDemos({ demos }: Props) {
  const [activeId, setActiveId] = useState<DemoMeta['id']>(demos[0].id);
  const active = useMemo(
    () => demos.find((d) => d.id === activeId) ?? demos[0],
    [activeId, demos],
  );
  const { trace, error } = useTrace(active.id);

  return (
    <div className={styles.wrap}>
      <div className={styles.cards} role="tablist" aria-label="Demos">
        {demos.map((d) => {
          const isActive = d.id === activeId;
          return (
            <button
              key={d.id}
              type="button"
              role="tab"
              aria-selected={isActive}
              aria-controls={`demo-panel-${d.id}`}
              className={clsx(styles.card, isActive && styles.cardActive)}
              onClick={() => setActiveId(d.id)}
            >
              <span className={styles.cardName}>{d.tab}</span>
              <span className={styles.cardMetric}>{d.metric}</span>
            </button>
          );
        })}
      </div>
      <div
        id={`demo-panel-${active.id}`}
        role="tabpanel"
        aria-labelledby={active.id}
        className={styles.panel}
      >
        {trace ? (
          <DemoPlayer trace={trace} key={trace.id} />
        ) : error ? (
          <div className={styles.placeholder}>could not load demo: {error}</div>
        ) : (
          <div className={styles.placeholder}>loading demo…</div>
        )}
      </div>
    </div>
  );
}
