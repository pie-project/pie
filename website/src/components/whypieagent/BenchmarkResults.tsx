import styles from './BenchmarkResults.module.css';

interface BenchmarkData {
  platform?: string;
  date?: string;
  rows: {
    scenario: string;
    standard: string;
    pie: string;
    improvement: string;
  }[];
}

interface Props {
  case1: { vllm_prefill_tokens: number; pie_prefill_tokens: number };
  case2: { vllm_reprefill_tokens: number; pie_reprefill_tokens: number; reduction_pct: number };
  benchmarks: BenchmarkData | null;
}

export default function BenchmarkResults({ case1, case2, benchmarks }: Props) {
  const rows = benchmarks?.rows ?? [
    {
      scenario: 'Cold start prefill',
      standard: `${case1.vllm_prefill_tokens.toLocaleString()} tokens`,
      pie: `${case1.pie_prefill_tokens.toLocaleString()} tokens`,
      improvement: `${Math.round((1 - case1.pie_prefill_tokens / case1.vllm_prefill_tokens) * 100)}%`,
    },
    {
      scenario: 'Channel switch re-prefill',
      standard: `${case2.vllm_reprefill_tokens.toLocaleString()} tokens`,
      pie: `${case2.pie_reprefill_tokens.toLocaleString()} tokens`,
      improvement: `${case2.reduction_pct}%`,
    },
  ];

  return (
    <div className={styles.wrapper}>
      <h3 className={styles.title}>Results</h3>

      <table className={styles.table}>
        <thead>
          <tr>
            <th>Scenario</th>
            <th>Standard Serving</th>
            <th>Pie</th>
            <th>Tokens saved</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const colonIdx = row.scenario.indexOf(':');
            const title = colonIdx >= 0 ? row.scenario.slice(0, colonIdx) : row.scenario;
            const rest = colonIdx >= 0 ? row.scenario.slice(colonIdx + 1).trim() : '';
            const renderMetric = (s: string) => {
              const lines = s.split('\n');
              return (
                <>
                  <span className={styles.metricPrimary}>{lines[0]}</span>
                  {lines[1] && <span className={styles.metricSecondary}>{lines[1]}</span>}
                </>
              );
            };
            return (
              <tr key={row.scenario}>
                <td>
                  <strong>{title}</strong>
                  {rest && <span className={styles.scenarioDesc}>{rest}</span>}
                </td>
                <td className={styles.monoCell}>{renderMetric(row.standard)}</td>
                <td className={styles.monoCell}>{renderMetric(row.pie)}</td>
                <td className={styles.monoCell}>{row.improvement}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <p className={styles.tableNote}>
        Percentages reflect token reduction (prefill for scenarios 1-3; input per turn for scenario 4).
        {benchmarks?.platform && <><br />Measured on {benchmarks.platform}.</>}
      </p>
    </div>
  );
}
