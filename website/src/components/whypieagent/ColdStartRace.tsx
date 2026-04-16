import { useEffect, useRef, useState, useCallback } from 'react';
import styles from './ColdStartRace.module.css';

interface Props {
  standardTokens: number;
  pieTokens: number;
}

const VLLM_DURATION = 3000; // ms
const PIE_DURATION = 150;   // ms

export default function ColdStartRace({ standardTokens, pieTokens }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [stdProgress, setVllmProgress] = useState(0); // 0..1
  const [pieProgress, setPieProgress] = useState(0);
  const [finished, setFinished] = useState(false);
  const hasAutoPlayed = useRef(false);
  const animRef = useRef<number | null>(null);

  const play = useCallback(() => {
    setVllmProgress(0);
    setPieProgress(0);
    setFinished(false);
    let start: number | null = null;

    const step = (ts: number) => {
      if (!start) start = ts;
      const elapsed = ts - start;
      const vP = Math.min(elapsed / VLLM_DURATION, 1);
      const pP = Math.min(elapsed / PIE_DURATION, 1);
      setVllmProgress(vP);
      setPieProgress(pP);
      if (vP < 1) {
        animRef.current = requestAnimationFrame(step);
      } else {
        setFinished(true);
      }
    };
    if (animRef.current) cancelAnimationFrame(animRef.current);
    animRef.current = requestAnimationFrame(step);
  }, []);

  useEffect(() => {
    if (!ref.current) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasAutoPlayed.current) {
          hasAutoPlayed.current = true;
          play();
        }
      },
      { threshold: 0.4 },
    );
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [play]);

  useEffect(() => {
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, []);

  const replay = useCallback(() => {
    setTimeout(() => play(), 50);
  }, [play]);

  const stdCurrent = Math.round(stdProgress * standardTokens);
  const pieCurrent = Math.round(pieProgress * pieTokens);
  const stdDone = stdProgress >= 1;
  const pieDone = pieProgress >= 1;

  const cn = (...classes: (string | false | undefined)[]) =>
    classes.filter(Boolean).join(' ');

  return (
    <div ref={ref} className={styles.outer}>
      <div className={styles.container}>
        {/* vLLM panel */}
        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <span className={cn(styles.dot, styles.dotRed)} />
            <span className={cn(styles.dot, styles.dotYellow)} />
            <span className={cn(styles.dot, styles.dotGreen)} />
            <span className={styles.panelTitle}>Standard Serving</span>
          </div>
          <div className={styles.panelBody}>
            <div className={styles.label}>Prefilling system prompt...</div>
            <div className={styles.progressTrack}>
              <div
                className={cn(styles.progressFill, styles.fillStd)}
                style={{ width: `${stdProgress * 100}%` }}
              />
            </div>
            <div className={styles.stats}>
              <span className={styles.tokens}>
                {stdCurrent.toLocaleString()}{' '}
                <span className={styles.tokenLabel}>tokens</span>
              </span>
              <span className={cn(styles.done, stdDone && styles.show)}>
                ready
              </span>
            </div>
          </div>
        </div>

        {/* Pie panel */}
        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <span className={cn(styles.dot, styles.dotRed)} />
            <span className={cn(styles.dot, styles.dotYellow)} />
            <span className={cn(styles.dot, styles.dotGreen)} />
            <span className={styles.panelTitle}>Pie inferlet</span>
          </div>
          <div className={styles.panelBody}>
            <div className={styles.label}>Loading cached KV state...</div>
            <div className={styles.progressTrack}>
              <div
                className={cn(styles.progressFill, styles.fillPie)}
                style={{ width: `${pieProgress * 100}%` }}
              />
            </div>
            <div className={styles.stats}>
              <span className={styles.tokens}>
                {pieCurrent.toLocaleString()}{' '}
                <span className={styles.tokenLabel}>tokens</span>
              </span>
              <span className={cn(styles.done, pieDone && styles.show)}>
                ready
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className={cn(styles.replayRow, finished && styles.replayVisible)}>
        <button className={styles.replayBtn} onClick={replay}>
          &#8635; Replay
        </button>
      </div>
    </div>
  );
}
