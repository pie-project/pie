import { useEffect, useRef, useState, useCallback } from 'react';
import styles from './MultiSessionTimeline.module.css';

interface Props {
  sessions?: number;
}

/*
  Vertical progress + memory view, side by side for Standard vs Pie.

  Timeline (top to bottom):
    Phase 1: Session 1 — prefill sys prompt + conversation
    Phase 2: Session 2 — arrives, evicts session 1
    Phase 3: Session 1 — returns, needs sys prompt again

  Memory view shows KV cache state at each phase:
    Standard: Session 2 evicts ALL of Session 1
    Pie: Session 2 evicts conversation but sys prompt stays (pinned)

  On Session 1 return:
    Standard: re-prefill sys prompt (slow)
    Pie: cache hit on sys prompt (instant)
*/

interface Phase {
  label: string;
  session: 1 | 2;
  type: 'sysprompt' | 'conversation' | 'reprefill' | 'import';
  /** Duration in time units */
  time: number;
}

const STD_PHASES: Phase[] = [
  { label: 'S1: prefill sys prompt',  session: 1, type: 'sysprompt',    time: 10 },
  { label: 'S1: conversation',        session: 1, type: 'conversation', time: 18 },
  { label: 'S2: prefill sys prompt',  session: 2, type: 'sysprompt',    time: 10 },
  { label: 'S2: conversation',        session: 2, type: 'conversation', time: 18 },
  { label: 'S1: re-prefill sys prompt', session: 1, type: 'reprefill', time: 10 },
  { label: 'S1: conversation',        session: 1, type: 'conversation', time: 14 },
];
// total = 80

const PIE_PHASES: Phase[] = [
  { label: 'S1: sys prompt (imported)', session: 1, type: 'import',       time: 3 },
  { label: 'S1: conversation',        session: 1, type: 'conversation', time: 18 },
  { label: 'S2: sys prompt (imported)', session: 2, type: 'import',     time: 3 },
  { label: 'S2: conversation',        session: 2, type: 'conversation', time: 18 },
  { label: 'S1: sys prompt (imported)', session: 1, type: 'import',     time: 3 },
  { label: 'S1: conversation',        session: 1, type: 'conversation', time: 14 },
];
// total = 59

const STD_TOTAL = STD_PHASES.reduce((s, p) => s + p.time, 0);
const PIE_TOTAL = PIE_PHASES.reduce((s, p) => s + p.time, 0); // 59
const ANIM_DURATION = 8000;

// Memory block state: fill = right edge (0..1), evict = left edge (0..1)
// fill=1, evict=0 → fully visible. fill=1, evict=0.5 → right half visible (evicting from left).
// fill=0.5, evict=0 → left half visible (filling from left).
interface MemBlock { fill: number; evict: number; }
interface MemState {
  s1Sys: MemBlock;
  s1Conv: MemBlock;
  s2Sys: MemBlock;
  s2Conv: MemBlock;
}
const EMPTY: MemBlock = { fill: 0, evict: 0 };
const FULL: MemBlock = { fill: 1, evict: 0 };

function getMemState(
  phases: Phase[],
  currentIdx: number,
  progress: number,
  isPie: boolean,
): MemState {
  // Pie: both sys prompts pre-warmed from start
  const state: MemState = isPie
    ? { s1Sys: { ...FULL }, s1Conv: { ...EMPTY }, s2Sys: { ...FULL }, s2Conv: { ...EMPTY } }
    : { s1Sys: { ...EMPTY }, s1Conv: { ...EMPTY }, s2Sys: { ...EMPTY }, s2Conv: { ...EMPTY } };

  for (let i = 0; i <= currentIdx && i < phases.length; i++) {
    const p = phases[i];
    const isActive = i === currentIdx;
    const t = isActive ? progress : 1;

    // S1 first round — sysprompt phase: sys fills gradually
    if (p.session === 1 && i < 4 && p.type === 'sysprompt' && !isPie) {
      state.s1Sys = isActive
        ? { fill: Math.min(1, t * 1.2), evict: 0 }
        : { fill: 1, evict: 0 };
    }
    // S1 first round — conversation phase: sys done, conv fills gradually
    if (p.session === 1 && i < 4 && p.type === 'conversation') {
      state.s1Sys = { fill: 1, evict: 0 };
      state.s1Conv = isActive
        ? { fill: Math.min(1, t * 1.5), evict: 0 }
        : { fill: 1, evict: 0 };
    }

    // S2 sysprompt phase: sys fills gradually (Standard only, Pie already has it)
    if (p.session === 2 && p.type === 'sysprompt' && !isPie) {
      state.s2Sys = isActive
        ? { fill: Math.min(1, t * 1.2), evict: 0 }
        : { fill: 1, evict: 0 };
    }
    // S2 import phase (Pie): sys already cached, just a quick import
    if (p.session === 2 && p.type === 'import') {
      // sys already full (pre-warmed), nothing to fill
    }

    // S2 conversation — S2 conv fills, then S1 eviction starts at t=0.3
    if (p.session === 2 && p.type === 'conversation') {
      state.s2Sys = { fill: 1, evict: 0 };
      state.s2Conv = isActive
        ? { fill: Math.min(1, t * 1.5), evict: 0 }
        : { fill: 1, evict: 0 };

      const evictT = Math.max(0, t - 0.3) / 0.7;
      if (isPie) {
        state.s1Conv = { fill: 1, evict: Math.min(1, evictT * 2) };
      } else {
        state.s1Sys = { fill: 1, evict: Math.min(1, evictT * 2.5) };
        state.s1Conv = { fill: 1, evict: Math.min(0.85, Math.max(0, evictT - 0.3) * 1.5) };
      }
    }

    // S1 re-prefill phase (Standard): evict old S1 conv sliver first,
    // then evict S2 sys to make room, while S1 sys fills
    if (i >= 4 && p.session === 1 && p.type === 'reprefill') {
      // S1 remaining conv sliver evicts first (oldest)
      state.s1Conv = { fill: 1, evict: Math.min(1, (isActive ? t : 1) * 3) };
      // S2 sys starts evicting as S1 sys needs the space
      const s2SysEvict = Math.min(1, Math.max(0, (isActive ? t : 1) - 0.2) * 2);
      state.s2Sys = { fill: 1, evict: s2SysEvict };
      // S1 sys fills gradually
      state.s1Sys = isActive
        ? { fill: Math.min(1, t * 1.2), evict: 0 }
        : { fill: 1, evict: 0 };
    }

    // S1 import phase (Pie, 3rd round): sys already cached, just evict S2 conv
    if (i >= 4 && p.session === 1 && p.type === 'import') {
      // Pie: S2 conv starts evicting to make room for S1 conv
      state.s2Conv = { fill: 1, evict: Math.min(0.5, (isActive ? t : 1) * 1) };
    }

    // S1 returns — conversation phase
    if (i >= 4 && p.session === 1 && p.type === 'conversation') {
      state.s1Sys = { fill: 1, evict: 0 };
      state.s1Conv = isActive
        ? { fill: Math.min(1, t * 1.5), evict: 0 }
        : { fill: 1, evict: 0 };

      // Continue evicting S2 as S1 conv grows
      const evictT = Math.max(0, t - 0.1) / 0.9;
      if (isPie) {
        // Pie: finish evicting S2 conv
        state.s2Conv = { fill: 1, evict: Math.min(1, 0.5 + evictT * 0.5) };
      } else {
        // Standard: S2 sys already evicted during reprefill. Now S2 conv evicts.
        state.s2Sys = { fill: 1, evict: 1 }; // fully gone
        state.s2Conv = { fill: 1, evict: Math.min(0.85, evictT * 1.5) };
      }
    }
  }
  return state;
}

// Map time to phase index and progress within phase
function timeToPhaseInfo(phases: Phase[], totalTime: number, clock: number) {
  const t = Math.min(clock, totalTime);
  let remaining = t;
  for (let i = 0; i < phases.length; i++) {
    if (remaining <= phases[i].time) {
      return { index: i, progress: remaining / phases[i].time };
    }
    remaining -= phases[i].time;
  }
  return { index: phases.length - 1, progress: 1 };
}

export default function MultiSessionTimeline(_props: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [clock, setClock] = useState(0); // 0..STD_TOTAL
  const [finished, setFinished] = useState(false);
  const hasAutoPlayed = useRef(false);
  const animRef = useRef<number | null>(null);

  const play = useCallback(() => {
    setClock(0);
    setFinished(false);
    let start: number | null = null;
    const step = (ts: number) => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / ANIM_DURATION, 1) * STD_TOTAL;
      setClock(p);
      if (p < STD_TOTAL) {
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
      { threshold: 0.3 },
    );
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [play]);

  useEffect(() => {
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, []);

  const stdInfo = timeToPhaseInfo(STD_PHASES, STD_TOTAL, clock);
  const pieInfo = timeToPhaseInfo(PIE_PHASES, PIE_TOTAL, Math.min(clock, PIE_TOTAL));
  const pieDone = clock >= PIE_TOTAL;

  const stdMem = getMemState(STD_PHASES, stdInfo.index, stdInfo.progress, false);
  const pieMem = getMemState(PIE_PHASES, pieInfo.index, pieInfo.progress, true);

  const cn = (...c: (string | false | undefined)[]) => c.filter(Boolean).join(' ');

  const phaseClass = (type: Phase['type'], session: 1 | 2) => {
    if (type === 'sysprompt') return session === 1 ? styles.phaseSysS1 : styles.phaseSysS2;
    if (type === 'conversation') return session === 1 ? styles.phaseConvS1 : styles.phaseConvS2;
    if (type === 'reprefill') return styles.phaseReprefillS1;
    if (type === 'import') return session === 1 ? styles.phaseSysS1 : styles.phaseSysS2;
    return '';
  };

  const sessionClass = (session: 1 | 2) => session === 1 ? styles.phaseS1 : styles.phaseS2;

  const renderTimeline = (phases: Phase[], total: number, info: { index: number; progress: number }, _done: boolean) => {
    // Height proportional to total time — Standard is taller, Pie is shorter
    const heightPx = Math.round((total / STD_TOTAL) * 700);
    return (
      <div className={styles.timeline} style={{ height: `${heightPx}px` }}>
        {phases.map((p, i) => {
          const heightPct = (p.time / total) * 100;
          const isPast = i < info.index || _done;
          const isActive = i === info.index && !_done;
          const isFuture = !isPast && !isActive;

          // Continuous fill: past=100%, active=partial, future=0%
          const fillPct = isPast ? 100 : isActive ? info.progress * 100 : 0;

          return (
            <div
              key={i}
              className={cn(styles.phase, sessionClass(p.session))}
              style={{ height: `${heightPct}%` }}
            >
              {/* Background (unfilled / future) */}
              <div className={styles.phaseBg} />
              {/* Fill overlay — grows from top */}
              <div
                className={cn(styles.phaseFill, phaseClass(p.type, p.session))}
                style={{ height: `${fillPct}%` }}
              />
              {/* Label — highlight (imported) in green */}
              <span className={cn(styles.phaseLabel, (isPast || isActive) && styles.phaseLabelVisible)}>
                {p.label.includes('(imported)')
                  ? <>{p.label.replace(' (imported)', '')}<span className={styles.importedTag}>&nbsp;(imported)</span></>
                  : p.label
                }
              </span>
              {p.type === 'import' && isPast && (
                <span className={styles.cacheHit}>cache hit!</span>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  const renderMemBlock = (block: MemBlock, type: 'sys' | 'conv', sessionClass: string) => {
    const hasContent = block.fill > 0.01 && block.evict < 0.99;
    return (
      <div
        className={cn(
          styles.memBlock,
          type === 'sys' ? styles.memBlockSys : styles.memBlockConv,
          sessionClass,
          hasContent && styles.memFilled,
        )}
        style={{
          '--mem-fill': `${Math.round(block.fill * 100)}%`,
          '--mem-evict': `${Math.round(block.evict * 100)}%`,
        } as React.CSSProperties}
      >
        <span className={styles.memBlockLabel}>{type === 'sys' ? 'sys' : 'conv'}</span>
      </div>
    );
  };

  const renderMemory = (mem: MemState, label: string) => (
    <div className={styles.memoryView}>
      <div className={styles.memTitle}>{label}</div>
      <div className={styles.memSlots}>
        <div className={cn(styles.memSlot, styles.memSlotS1)}>
          <div className={styles.memSlotLabel}>S1</div>
          {renderMemBlock(mem.s1Sys, 'sys', styles.memS1Sys)}
          {renderMemBlock(mem.s1Conv, 'conv', styles.memS1Conv)}
        </div>
        <div className={cn(styles.memSlot, styles.memSlotS2)}>
          <div className={styles.memSlotLabel}>S2</div>
          {renderMemBlock(mem.s2Sys, 'sys', styles.memS2Sys)}
          {renderMemBlock(mem.s2Conv, 'conv', styles.memS2Conv)}
        </div>
      </div>
    </div>
  );

  return (
    <div ref={ref} className={styles.wrapper}>
      {/* Memory views side by side */}
      <div className={styles.memRow}>
        <div className={styles.memCol}>
          <h4 className={styles.systemTitle}>Standard Serving</h4>
          {renderMemory(stdMem, 'KV Cache')}
        </div>
        <div className={styles.memColSpacer} />
        <div className={styles.memCol}>
          <h4 className={styles.systemTitle}>Pie</h4>
          {renderMemory(pieMem, 'KV Cache')}
        </div>
      </div>

      {/* Progress timelines with time arrow in between, bottom-aligned */}
      <div className={styles.progressRow}>
        <div className={styles.progressCol}>
          <div className={styles.sectionTitle}>Progress</div>
          {renderTimeline(STD_PHASES, STD_TOTAL, stdInfo, false)}
        </div>

        <div className={styles.timeArrowCol}>
          <div className={styles.timeArrow}>
            <span>time</span>
            <div className={styles.timeArrowLine} />
          </div>
        </div>

        <div className={styles.progressCol}>
          <div className={styles.sectionTitle}>Progress</div>
          {renderTimeline(PIE_PHASES, PIE_TOTAL, pieInfo, pieDone)}
        </div>
      </div>

      {/* Legend */}
      <div className={styles.legend}>
        <div className={styles.legendItem}>
          <span className={cn(styles.legendDot, styles.dotSys)} /> System prompt
        </div>
        <div className={styles.legendItem}>
          <span className={cn(styles.legendDot, styles.dotConv)} /> Conversation
        </div>
        <div className={styles.legendItem}>
          <span className={cn(styles.legendDot, styles.dotReprefill)} /> Re-prefill
        </div>
        <div className={styles.legendItem}>
          <span className={cn(styles.legendDot, styles.dotImport)} /> KV import
        </div>
      </div>

      {/* &#8635; Replay */}
      <div className={cn(styles.replayRow, finished && styles.replayVisible)}>
        <button className={styles.replayBtn} onClick={play}>&#8635; Replay</button>
      </div>
    </div>
  );
}
