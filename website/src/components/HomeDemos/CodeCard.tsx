import CodeBlock from '@theme/CodeBlock';
import styles from './CodeCard.module.css';

interface Props {
  title: string;
  description: string;
  language: string;
  code: string;
  callout?: string;
}

export default function CodeCard({ title, description, language, code, callout }: Props) {
  return (
    <div className={styles.card}>
      <div className={styles.head}>
        <h3 className={styles.title}>{title}</h3>
        <p className={styles.description}>{description}</p>
      </div>
      <div className={styles.codeWrap}>
        <CodeBlock language={language}>{code}</CodeBlock>
      </div>
      {callout ? <p className={styles.callout}>{callout}</p> : null}
    </div>
  );
}
