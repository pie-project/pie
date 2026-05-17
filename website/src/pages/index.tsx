import { type ReactNode } from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import ThemedImage from '@theme/ThemedImage';
import useBaseUrl from '@docusaurus/useBaseUrl';
import HomeDemos, { type DemoMeta } from '@site/src/components/HomeDemos';
import styles from './index.module.css';

const FORWARD_PASS_DEMOS: DemoMeta[] = [
    {
        id: 'custom-spec',
        tab: 'Custom speculative decoding',
        metric: 'Drafter ships with the request',
    },
    {
        id: 'grammar',
        tab: 'Custom logit masking',
        metric: 'JSON Schema or custom logit mask',
    },
    {
        id: 'watermark',
        tab: 'Custom sampling',
        metric: 'Sampler bias and in-process detector',
    },
];

const KV_DEMOS: DemoMeta[] = [
    {
        id: 'self-correct',
        tab: 'Cache rewind',
        metric: 'User code rewinds the cache',
    },
    {
        id: 'parallel-fork',
        tab: 'Prefix sharing',
        metric: 'fork() shares the prompt',
    },
    {
        id: 'persistent-kv',
        tab: 'Cache persistence',
        metric: 'Cache is the session',
    },
];

const IO_DEMOS: DemoMeta[] = [
    {
        id: 'mcp-tools',
        tab: 'Tools',
        metric: 'Tool loop in user code',
    },
    {
        id: 'filesystem',
        tab: 'Virtual filesystem',
        metric: 'std::fs to persistent /scratch',
    },
    {
        id: 'messaging',
        tab: 'Messaging',
        metric: 'Topics inside the engine',
    },
];

function Hero() {
    return (
        <header className={clsx('hero', styles.heroBanner)}>
            <div className="container">
                <h1 className={styles.heroTitle}>
                    A <span>programmable</span> LLM serving system.
                </h1>
                <p className={styles.heroSubtitle}>
                    High-performance inference engine where you write the loop. <br />
                    Forward passes are library calls in your inferlet.
                </p>
                <div className={styles.heroLinks}>
                    <Link className="button button--primary button--lg" to="/docs/overview/what-is-pie">
                        What is Pie?
                    </Link>
                    <Link className="button button--secondary button--lg" to="/docs/guide/install">
                        Get started
                    </Link>
                    <Link className="button button--secondary button--lg" to="https://github.com/pie-project/pie">
                        GitHub
                    </Link>
                </div>
            </div>
        </header>
    );
}

function ServingLoop() {
    return (
        <section className={styles.section}>
            <div className="container">
                <h2 className={styles.sectionTitle}>Serve programs, not prompts</h2>
                <p className={styles.sectionSubtitle}>
                    In existing serving systems, the inference workflow is baked into the engine.
                    In Pie, you write it.
                </p>
                <div className={styles.figuresGrid}>
                    <div className={styles.figureCard}>
                        <h3>Conventional serving systems</h3>
                        <ThemedImage
                            sources={{
                                light: useBaseUrl('/img/current-serving.svg'),
                                dark: useBaseUrl('/img/current-serving-dark.svg'),
                            }}
                            alt="A conventional serving system. Prompts from users enter the engine and pass through a fixed pipeline of batch, embed, prefill or decode, and sample stages, with one global autoregressive loop."
                        />
                        <p>
                            Every request runs through the same fixed pipeline.
                            Branching and tool calls live outside the engine.
                        </p>
                    </div>
                    <div className={styles.figureCard}>
                        <h3>Programmable serving system - Pie</h3>
                        <ThemedImage
                            sources={{
                                light: useBaseUrl('/img/programmable-serving.svg'),
                                dark: useBaseUrl('/img/programmable-serving-dark.svg'),
                            }}
                            alt="Pie's serving model. Each application runs as an inferlet inside the engine, calling into the model's KV cache and forward pass through a control layer."
                        />
                        <p>
                            Each inferlet runs its own workflow inside the engine.
                            It controls the KV cache, forward pass, and tool calls directly.
                        </p>
                    </div>
                </div>
            </div>
        </section>
    );
}

interface ThemeSectionProps {
    eyebrow: string;
    heading: string;
    subtitle: string;
    sectionClassName?: string;
    children: ReactNode;
}

function ThemeSection({ eyebrow, heading, subtitle, sectionClassName, children }: ThemeSectionProps) {
    return (
        <section className={clsx(styles.section, styles.themeSection, sectionClassName)}>
            <div className="container">
                <p className={styles.themeEyebrow}>{eyebrow}</p>
                <h2 className={styles.themeHeading}>{heading}</h2>
                <p className={styles.themeSubtitle}>{subtitle}</p>
                {children}
            </div>
        </section>
    );
}

function KvCacheTheme() {
    return (
        <ThemeSection
            eyebrow="01"
            heading="Program the KV cache"
            subtitle="Branch, rewind, persist, and reopen the cache by calling engine APIs from user code."
        >
            <HomeDemos demos={KV_DEMOS} />
        </ThemeSection>
    );
}

function ForwardPassTheme() {
    return (
        <ThemeSection
            eyebrow="02"
            heading="Program the forward pass"
            subtitle="The decoder loop, sampler, and constraint matcher are exposed to user code. Write your own drafter, sampling rule, or grammar."
        >
            <HomeDemos demos={FORWARD_PASS_DEMOS} />
        </ThemeSection>
    );
}

function IoTheme() {
    return (
        <ThemeSection
            eyebrow="03"
            heading="Program the I/O"
            subtitle="Tool clients, persistent files, and message brokers run inside the engine. User code calls them like a library."
        >
            <HomeDemos demos={IO_DEMOS} />
        </ThemeSection>
    );
}

function ClosingCta() {
    return (
        <section className={clsx('hero', styles.heroBanner, styles.closingCta)} aria-labelledby="closing-cta-heading">
            <div className="container">
                <h2 id="closing-cta-heading" className={styles.heroTitle}>
                    Ready to <span>write the loop?</span>
                </h2>
                <p className={styles.heroSubtitle}>
                    Install the engine, ship your first inferlet, and program the forward pass, KV cache, and I/O from your own code.
                </p>
                <div className={styles.heroLinks}>
                    <Link className="button button--primary button--lg" to="/docs/guide/install">
                        Get started
                    </Link>
                    <Link className="button button--secondary button--lg" to="/docs/overview/what-is-pie">
                        What is Pie?
                    </Link>
                    <Link className="button button--secondary button--lg" to="https://github.com/pie-project/pie">
                        GitHub
                    </Link>
                </div>
            </div>
        </section>
    );
}

export default function Home(): ReactNode {
    return (
        <Layout
            title="Programmable LLM serving"
            description="Pie is a programmable serving system for LLM inference."
        >
            <Hero />
            <main>
                <ServingLoop />
                <KvCacheTheme />
                <ForwardPassTheme />
                <IoTheme />
                <ClosingCta />
            </main>
        </Layout>
    );
}
