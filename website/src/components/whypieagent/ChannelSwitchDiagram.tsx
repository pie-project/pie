import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import styles from './ChannelSwitchDiagram.module.css';

interface Section {
  name: string;
  token_start: number;
  token_end: number;
  tokens: number;
  stability: 'static' | 'dynamic';
}

interface Props {
  sections: Section[];
  totalTokens: number;
  divergenceOriginal: number;
  divergenceReordered: number;
  standardReprefill: number;
  pieReprefill: number;
  reductionPct: number;
}

type BlockState = 'default' | 'cached' | 'invalidated' | 'changed';

function shortName(name: string): string {
  let s = name.replace(/^#{1,3}\s+/, '');
  if (s.includes(' + ')) s = s.split(' + ')[0];
  s = s.replace(/\/home\/node\/.openclaw\/workspace\//, '');
  s = s.replace(/[\u{1F300}-\u{1FAFF}]/gu, '').trim();
  if (s.length > 24) s = s.slice(0, 22) + '\u2026';
  return s;
}

export default function ChannelSwitchDiagram({
  sections,
  totalTokens,
  divergenceOriginal,
  standardReprefill,
  pieReprefill,
  reductionPct,
}: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [switched, setSwitched] = useState(false);
  const [stdRevealed, setStdRevealed] = useState(0);
  const [pieRevealed, setPieRevealed] = useState(0);
  const [finished, setFinished] = useState(false);
  const hasAutoPlayed = useRef(false);
  const stdTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pieTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const originalBlocks = useMemo(
    () => sections.map((s) => ({ ...s, shortName: shortName(s.name) })),
    [sections],
  );

  const reorderedBlocks = useMemo(() => {
    const statics = sections.filter((s) => s.stability === 'static');
    const dynamics = sections.filter((s) => s.stability === 'dynamic');
    return [...statics, ...dynamics].map((s) => ({ ...s, shortName: shortName(s.name) }));
  }, [sections]);

  // Compute target states (what each block SHOULD be when revealed)
  const originalTargets: BlockState[] = useMemo(() => {
    if (!switched) return originalBlocks.map(() => 'default');
    return originalBlocks.map((b) => {
      const mid = b.token_start + b.tokens / 2;
      if (b.stability === 'dynamic') return 'changed';
      if (mid < divergenceOriginal) return 'cached';
      return 'invalidated';
    });
  }, [originalBlocks, switched, divergenceOriginal]);

  const reorderedTargets: BlockState[] = useMemo(() => {
    if (!switched) return reorderedBlocks.map(() => 'default');
    return reorderedBlocks.map((b) => (b.stability === 'dynamic' ? 'changed' : 'cached'));
  }, [reorderedBlocks, switched]);

  // Two independent reveal chains — each side advances at its own pace
  // Green (cached) = 100ms, orange/red (invalidated/changed) = 350ms
  const FAST = 100;
  const SLOW = 350;

  const delayFor = (state: BlockState) => state === 'cached' ? FAST : SLOW;

  // Standard side reveal
  useEffect(() => {
    if (!switched) {
      setStdRevealed(0);
      if (stdTimerRef.current) clearTimeout(stdTimerRef.current);
      return;
    }
    let step = 0;
    const advance = () => {
      step++;
      setStdRevealed(step);
      if (step < originalBlocks.length) {
        stdTimerRef.current = setTimeout(advance, delayFor(originalTargets[step]));
      }
    };
    stdTimerRef.current = setTimeout(advance, delayFor(originalTargets[0]));
    return () => { if (stdTimerRef.current) clearTimeout(stdTimerRef.current); };
  }, [switched, originalBlocks.length, originalTargets]);

  // Pie side reveal
  useEffect(() => {
    if (!switched) {
      setPieRevealed(0);
      if (pieTimerRef.current) clearTimeout(pieTimerRef.current);
      return;
    }
    let step = 0;
    const advance = () => {
      step++;
      setPieRevealed(step);
      if (step < reorderedBlocks.length) {
        pieTimerRef.current = setTimeout(advance, delayFor(reorderedTargets[step]));
      }
    };
    pieTimerRef.current = setTimeout(advance, delayFor(reorderedTargets[0]));
    return () => { if (pieTimerRef.current) clearTimeout(pieTimerRef.current); };
  }, [switched, reorderedBlocks.length, reorderedTargets]);

  // Finished when BOTH sides are done
  const stdDone = stdRevealed >= originalBlocks.length;
  const pieDone = pieRevealed >= reorderedBlocks.length;
  useEffect(() => {
    if (switched && stdDone && pieDone) {
      const t = setTimeout(() => setFinished(true), 400);
      return () => clearTimeout(t);
    }
  }, [switched, stdDone, pieDone]);

  const play = useCallback(() => {
    setSwitched(false);
    setStdRevealed(0);
    setPieRevealed(0);
    setFinished(false);
    setTimeout(() => setSwitched(true), 100);
  }, []);

  // Auto-play on scroll into view
  useEffect(() => {
    if (!ref.current) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasAutoPlayed.current) {
          hasAutoPlayed.current = true;
          setTimeout(() => setSwitched(true), 800);
        }
      },
      { threshold: 0.35 },
    );
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  const overlayClass = (state: BlockState) => {
    switch (state) {
      case 'cached': return styles.overlayCached;
      case 'invalidated': return styles.overlayInvalidated;
      case 'changed': return styles.overlayChanged;
      default: return '';
    }
  };

  const heightForTokens = (tokens: number) =>
    Math.max(18, Math.round((tokens / totalTokens) * 480));

  const cn = (...classes: (string | false | undefined)[]) =>
    classes.filter(Boolean).join(' ');

  const cachedOriginal = stdDone
    ? originalBlocks.reduce((sum, b, i) => sum + (originalTargets[i] === 'cached' ? b.tokens : 0), 0)
    : 0;
  const cachedReordered = pieDone
    ? reorderedBlocks.reduce((sum, b, i) => sum + (reorderedTargets[i] === 'cached' ? b.tokens : 0), 0)
    : 0;

  return (
    <div ref={ref} className={styles.wrapper}>
      {/* Side by side diagrams */}
      <div className={styles.sideBySide}>
        {/* Standard side */}
        <div className={styles.column}>
          <h4 className={styles.columnTitle}>Standard Serving (original layout)</h4>
          <div className={styles.blockStack}>
            {originalBlocks.map((b, i) => {
              const revealed = i < stdRevealed;
              const target = originalTargets[i];
              return (
                <div
                  key={`orig-${i}`}
                  className={styles.block}
                  style={{ height: heightForTokens(b.tokens) }}
                  title={`${b.shortName} (${b.tokens} tok) — ${b.stability}`}
                >
                  <div className={cn(styles.blockOverlay, revealed && overlayClass(target), revealed && styles.blockWiped)} />
                  <span className={styles.blockLabel}>{b.shortName}</span>
                  <span className={styles.blockTokens}>{b.tokens}</span>
                </div>
              );
            })}
          </div>
          {stdDone && switched && (
            <div className={styles.stats}>
              <span className={styles.statCached}>
                {cachedOriginal.toLocaleString()} cached
              </span>
              <span className={styles.statReprefill}>
                {standardReprefill.toLocaleString()} re-prefilled
              </span>
            </div>
          )}
        </div>

        {/* Pie side */}
        <div className={styles.column}>
          <h4 className={styles.columnTitle}>Pie (reordered layout)</h4>
          <div className={styles.blockStack}>
            {reorderedBlocks.map((b, i) => {
              const revealed = i < pieRevealed;
              const target = reorderedTargets[i];
              return (
                <div
                  key={`reord-${i}`}
                  className={styles.block}
                  style={{ height: heightForTokens(b.tokens) }}
                  title={`${b.shortName} (${b.tokens} tok) — ${b.stability}`}
                >
                  <div className={cn(styles.blockOverlay, revealed && overlayClass(target), revealed && styles.blockWiped)} />
                  <span className={styles.blockLabel}>{b.shortName}</span>
                  <span className={styles.blockTokens}>{b.tokens}</span>
                </div>
              );
            })}
          </div>
          {pieDone && switched && (
            <div className={styles.stats}>
              <span className={styles.statCached}>
                {cachedReordered.toLocaleString()} cached
              </span>
              <span className={styles.statReprefill}>
                {pieReprefill.toLocaleString()} re-prefilled
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Legend */}
      <div className={styles.legend}>
        <div className={styles.legendItem}>
          <span className={cn(styles.legendDot, styles.dotCached)} />
          <span>Cached</span>
        </div>
        <div className={styles.legendItem}>
          <span className={cn(styles.legendDot, styles.dotChanged)} />
          <span>Changed</span>
        </div>
        <div className={styles.legendItem}>
          <span className={cn(styles.legendDot, styles.dotInvalidated)} />
          <span>Cascade invalidated</span>
        </div>
      </div>

      {/* Replay */}
      <div className={cn(styles.replayRow, finished && styles.replayVisible)}>
        <button className={styles.replayBtn} onClick={play}>&#8635; Replay</button>
      </div>

      {/* Summary — below replay */}
      {stdDone && pieDone && switched && (
        <div className={styles.summaryBar}>
          <span className={styles.summaryLabel}>Tokens needing recomputation</span>
          <span className={styles.summaryDetail}>
            {standardReprefill.toLocaleString()} &rarr; {pieReprefill.toLocaleString()}
          </span>
          <span className={styles.summaryValue}>{reductionPct}% less recomputation</span>
        </div>
      )}
    </div>
  );
}
