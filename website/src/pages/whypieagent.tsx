import { useEffect, useState } from 'react';
import Layout from '@theme/Layout';
import useBaseUrl from '@docusaurus/useBaseUrl';
import BrowserOnly from '@docusaurus/BrowserOnly';
import styles from './whypieagent.module.css';
import ColdStartRace from '@site/src/components/whypieagent/ColdStartRace';
import ChannelSwitchDiagram from '@site/src/components/whypieagent/ChannelSwitchDiagram';
import MultiSessionTimeline from '@site/src/components/whypieagent/MultiSessionTimeline';
import BenchmarkResults from '@site/src/components/whypieagent/BenchmarkResults';

interface DemoData {
  total_tokens: number;
  sections: Array<{
    name: string;
    token_start: number;
    token_end: number;
    tokens: number;
    stability: 'static' | 'dynamic';
  }>;
  divergence_original: number;
  divergence_reordered: number;
  case1: {
    vllm_prefill_tokens: number;
    pie_prefill_tokens: number;
  };
  case2: {
    vllm_reprefill_tokens: number;
    pie_reprefill_tokens: number;
    reduction_pct: number;
  };
  case3: {
    sessions: number;
  };
  benchmarks: null | {
    platform?: string;
    date?: string;
    rows: Array<{
      scenario: string;
      standard: string;
      pie: string;
      improvement: string;
    }>;
  };
}

function WhyPieAgentContent() {
  const dataUrl = useBaseUrl('/data/demo-data.json');
  const [data, setData] = useState<DemoData | null>(null);

  useEffect(() => {
    fetch(dataUrl)
      .then((r) => r.json())
      .then(setData)
      .catch(console.error);
  }, [dataUrl]);

  if (!data) {
    return (
      <div className={styles.loading}>
        <span>Loading...</span>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <section className={styles.hero}>
        <h1 className={styles.heroTitle}>
          Smarter KV Caching for <span className={styles.accent}>AI Agents</span>
        </h1>
        <p className={styles.heroSubtitle}>
          Agentic workloads switch contexts, juggle sessions, and restart often.
          Pie keeps the right data in cache so your model spends less time
          re-computing what it already knows.
        </p>
      </section>

      <main>
        <section className={styles.caseSection}>
          <div className={styles.container}>
            <div className={styles.caseHeader}>
              <p className={styles.caseLabel}>Case 1</p>
              <h2 className={styles.caseTitle}>Cold Start</h2>
              <p className={styles.caseDescription}>
                The system prompt is already cached before the first request arrives.
              </p>
            </div>

            <ColdStartRace
              standardTokens={data.case1.vllm_prefill_tokens}
              pieTokens={data.case1.pie_prefill_tokens}
            />
          </div>
        </section>

        <section className={styles.caseSectionAlt}>
          <div className={styles.container}>
            <div className={styles.caseHeader}>
              <p className={styles.caseLabel}>Case 2</p>
              <h2 className={styles.caseTitle}>Channel Switch</h2>
              <p className={styles.caseDescription}>
                A small change in the middle of the prompt invalidates everything after it.
                Reordering the prompt keeps the cache stable.
              </p>
            </div>

            <ChannelSwitchDiagram
              sections={data.sections}
              totalTokens={data.total_tokens}
              divergenceOriginal={data.divergence_original}
              divergenceReordered={data.divergence_reordered}
              standardReprefill={data.case2.vllm_reprefill_tokens}
              pieReprefill={data.case2.pie_reprefill_tokens}
              reductionPct={data.case2.reduction_pct}
            />

            <p className={styles.fairNote}>
              <strong>Note:</strong> the upstream application could reorder
              its prompt to achieve the same effect. Pie's inferlet does this
              transparently without requiring upstream changes.
            </p>
          </div>
        </section>

        <section className={styles.caseSection}>
          <div className={styles.container}>
            <div className={styles.caseHeader}>
              <p className={styles.caseLabel}>Case 3</p>
              <h2 className={styles.caseTitle}>Session Resume Under Memory Pressure</h2>
              <p className={styles.caseDescription}>
                When sessions compete for GPU memory, idle KV gets evicted.
                Pie pins the system prompt so it never needs re-prefilling.
              </p>
            </div>

            <MultiSessionTimeline />
          </div>
        </section>

        <section className={styles.caseSectionAlt}>
          <div className={styles.container}>
            <BenchmarkResults
              case1={data.case1}
              case2={data.case2}
              benchmarks={data.benchmarks}
            />
          </div>
        </section>

        <section className={styles.disclaimerSection}>
          <div className={styles.container}>
            <p className={styles.disclaimer}>
              The visualizations on this page are illustrative diagrams designed
              to aid understanding of structural differences between serving
              approaches. They do not represent exact measurements or benchmarks.
              Actual performance varies by model, hardware, workload, and
              configuration. Token counts are derived from prompt structure
              analysis of a real agentic system prompt. Timing proportions in
              Case 3 are schematic. Measured benchmarks will be published
              separately.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}

export default function WhyPieAgent(): JSX.Element {
  return (
    <Layout
      title="Why Pie for Agents"
      description="Smarter KV caching for AI agents: how Pie handles cold start, channel switch, and multi-session workloads."
    >
      <BrowserOnly>{() => <WhyPieAgentContent />}</BrowserOnly>
    </Layout>
  );
}
