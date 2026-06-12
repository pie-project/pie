import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import clsx from 'clsx';
import CodeBlock from '@theme/CodeBlock';
import type { DemoEvent, DemoPane, DemoTrace } from './types';
import styles from './DemoPlayer.module.css';

interface Props {
  trace: DemoTrace;
}

function paneEnd(pane: DemoPane): number {
  if (pane.events.length === 0) return 0;
  return pane.events[pane.events.length - 1].t;
}

function classifyEventBlock(kind: DemoEvent['kind']): boolean {
  return (
    kind === 'info' ||
    kind === 'section' ||
    kind === 'good' ||
    kind === 'bad' ||
    kind === 'rewind' ||
    kind === 'result'
  );
}

const FORK_CLASSES: Record<string, string> = {
  A: styles.forkA,
  B: styles.forkB,
  C: styles.forkC,
  D: styles.forkD,
  '*': styles.forkAll,
};

function renderEvent(ev: DemoEvent, key: number) {
  switch (ev.kind) {
    case 'text':
      return (
        <span key={key} className={styles.text}>
          {ev.value}
        </span>
      );
    case 'burst':
      return (
        <span key={key} className={styles.burst}>
          {ev.value}
        </span>
      );
    case 'stream': {
      const cls = ev.fork ? FORK_CLASSES[ev.fork] ?? styles.text : styles.text;
      return (
        <span key={key} className={cls}>
          {ev.value}
        </span>
      );
    }
    case 'info':
      return (
        <div key={key} className={styles.info}>
          <span className={styles.infoMark}>›</span> {ev.value}
        </div>
      );
    case 'section':
      return (
        <div key={key} className={styles.section}>
          {ev.value}
        </div>
      );
    case 'good':
      return (
        <div key={key} className={styles.good}>
          <span className={styles.goodMark}>✓</span> {ev.value}
        </div>
      );
    case 'bad':
      return (
        <div key={key} className={styles.bad}>
          <span className={styles.badMark}>✗</span> {ev.value}
        </div>
      );
    case 'rewind':
      return (
        <div key={key} className={styles.rewind}>
          <span className={styles.rewindMark}>↺</span> {ev.value}
        </div>
      );
    case 'result':
      return (
        <div key={key} className={styles.result}>
          {ev.value}
        </div>
      );
    default:
      return null;
  }
}

function Pane({
  pane,
  elapsedMs,
  isComplete,
  onReplay,
}: {
  pane: DemoPane;
  elapsedMs: number;
  isComplete: boolean;
  onReplay: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const visible = useMemo(
    () => pane.events.filter((e) => e.t <= elapsedMs),
    [pane.events, elapsedMs],
  );

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [visible.length]);

  return (
    <div className={clsx(styles.pane, styles[`pane_${pane.tone}`])}>
      <div className={styles.paneHeader}>
        <span className={styles.paneLabel}>{pane.label}</span>
        {pane.note ? <span className={styles.paneNote}>{pane.note}</span> : null}
      </div>
      <div ref={ref} className={styles.paneBody}>
        {visible.map((ev, i) => renderEvent(ev, i))}
        <span className={styles.cursor} aria-hidden>▍</span>
      </div>
      <div
        className={clsx(styles.paneFooter, isComplete && styles.paneFooterVisible)}
        aria-hidden={!isComplete}
      >
        <button
          type="button"
          className={styles.paneReplay}
          onClick={onReplay}
          tabIndex={isComplete ? 0 : -1}
          aria-label="Replay animation"
        >
          ↻ Replay
        </button>
      </div>
    </div>
  );
}

export default function DemoPlayer({ trace }: Props) {
  const [elapsedMs, setElapsedMs] = useState(0);
  const [running, setRunning] = useState(true);
  const [codeOpen, setCodeOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const startRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);
  const pausedAtRef = useRef<number | null>(null);
  const hasStartedRef = useRef(false);

  const totalMs = useMemo(
    () => Math.max(paneEnd(trace.naive), paneEnd(trace.pie)) + 800,
    [trace],
  );

  const stopRaf = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const tick = useCallback(
    (ts: number) => {
      if (startRef.current == null) startRef.current = ts;
      const elapsed = ts - startRef.current;
      if (elapsed >= totalMs) {
        setElapsedMs(totalMs);
        rafRef.current = null;
        pausedAtRef.current = totalMs;
        setRunning(false);
        return;
      }
      setElapsedMs(elapsed);
      rafRef.current = requestAnimationFrame(tick);
    },
    [totalMs],
  );

  const play = useCallback(() => {
    stopRaf();
    setRunning(true);
    startRef.current = null;
    setElapsedMs(0);
    rafRef.current = requestAnimationFrame(tick);
  }, [stopRaf, tick]);

  const pause = useCallback(() => {
    stopRaf();
    setRunning(false);
    pausedAtRef.current = elapsedMs;
  }, [stopRaf, elapsedMs]);

  const resume = useCallback(() => {
    if (running) return;
    setRunning(true);
    const resumeFrom = pausedAtRef.current ?? 0;
    startRef.current = null;
    const startTick = (ts: number) => {
      startRef.current = ts - resumeFrom;
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(startTick);
  }, [running, tick]);

  // Latest-callback refs so the IntersectionObserver below always reads
  // current state instead of the closure from its setup render. Without
  // this, after completion the observer's `running` view is stuck at
  // its initial value and any intersection flicker during/after Replay
  // strands the animation mid-frame.
  const runningRef = useRef(running);
  const playRef = useRef(play);
  const pauseRef = useRef(pause);
  const resumeRef = useRef(resume);
  useEffect(() => {
    runningRef.current = running;
    playRef.current = play;
    pauseRef.current = pause;
    resumeRef.current = resume;
  });

  // Auto-play when scrolled into view; pause when leaving.
  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          if (!hasStartedRef.current) {
            hasStartedRef.current = true;
            playRef.current();
          } else if (!runningRef.current) {
            resumeRef.current();
          }
        } else if (runningRef.current) {
          pauseRef.current();
        }
      },
      { threshold: 0.3 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [trace.id]);

  // Reset when trace changes.
  useEffect(() => {
    hasStartedRef.current = false;
    setCodeOpen(false);
    play();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trace.id]);

  // Cleanup on unmount.
  useEffect(() => {
    return () => stopRaf();
  }, [stopRaf]);

  const inferletDir = `demo-${trace.id}`;
  const inferletUrl = `https://github.com/pie-project/pie/tree/main/inferlets/${inferletDir}`;

  return (
    <div ref={containerRef} className={styles.player}>
      <div className={styles.head}>
        <div className={styles.headText}>
          <h3 className={styles.title}>{trace.title}</h3>
          <p className={styles.tagline}>{trace.tagline}</p>
        </div>
      </div>
      <div className={styles.question}>
        <div className={styles.cardHeader}>
          <span className={styles.glyph} aria-hidden>1</span>
          <span className={styles.questionLabel}>Task</span>
        </div>
        <p className={styles.questionText}>{trace.question}</p>
      </div>
      <div className={styles.panes}>
        <Pane
          pane={trace.naive}
          elapsedMs={elapsedMs}
          isComplete={elapsedMs >= totalMs}
          onReplay={play}
        />
        <Pane
          pane={trace.pie}
          elapsedMs={elapsedMs}
          isComplete={elapsedMs >= totalMs}
          onReplay={play}
        />
      </div>
      {trace.code ? (
        <div className={styles.codeWrap}>
          <button
            type="button"
            className={clsx(styles.codeToggle, codeOpen && styles.codeToggleOpen)}
            onClick={() => setCodeOpen((v) => !v)}
            aria-expanded={codeOpen}
            aria-controls={`demo-code-${trace.id}`}
          >
            <span className={styles.glyph} aria-hidden>2</span>
            <span className={styles.codeToggleLabel}>
              {codeOpen ? 'Hide the code' : 'Read the code'}
            </span>
            <span className={styles.codeToggleCaret} aria-hidden>
              {codeOpen ? '▾' : '▸'}
            </span>
          </button>
          {codeOpen ? (
            <div id={`demo-code-${trace.id}`} className={styles.codeRow}>
              <div className={clsx(styles.codeColumn, styles.codeColumn_warn)}>
                <div className={styles.codeHeader}>
                  <span className={styles.codeLabel}>Stock API</span>
                </div>
                <CodeBlock language={trace.code.naive.language}>
                  {trace.code.naive.value}
                </CodeBlock>
              </div>
              <div className={clsx(styles.codeColumn, styles.codeColumn_good)}>
                <div className={styles.codeHeader}>
                  <span className={styles.codeLabel}>Pie inferlet</span>
                  <a
                    className={styles.codeSource}
                    href={inferletUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    inferlets/{inferletDir} ↗
                  </a>
                </div>
                <CodeBlock language={trace.code.pie.language}>
                  {trace.code.pie.value}
                </CodeBlock>
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
      {trace.runCommand ? (
        <div className={styles.tryIt}>
          <div className={styles.cardHeader}>
            <span className={styles.glyph} aria-hidden>3</span>
            <span className={styles.tryItLabel}>Try it yourself</span>
          </div>
          <CodeBlock language="bash">{trace.runCommand}</CodeBlock>
        </div>
      ) : null}
    </div>
  );
}
