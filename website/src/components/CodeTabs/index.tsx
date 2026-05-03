// MDX-friendly code-tabs that mirror the homepage `<Sample>` pattern
// (`src/pages/index.tsx`). Use when Docusaurus's stock `<Tabs>` —
// designed for arbitrary content — produces a visible seam between the
// tab strip and the code panel; this component renders the tabs as a
// header bar of the same panel surface.
//
// Usage in MDX:
//
//   import { CodeTabs, CodeTabItem } from '@site/src/components/CodeTabs';
//
//   <CodeTabs>
//     <CodeTabItem value="rust" label="Rust">
//       ```rust
//       fn main() {}
//       ```
//     </CodeTabItem>
//     <CodeTabItem value="py" label="Python">
//       ```python
//       def main(): pass
//       ```
//     </CodeTabItem>
//   </CodeTabs>

import { Children, isValidElement, useState } from 'react';
import type { ReactElement, ReactNode } from 'react';
import clsx from 'clsx';

import styles from './styles.module.css';

interface CodeTabItemProps {
    label: string;
    value: string;
    children: ReactNode;
}

export function CodeTabItem({ children }: CodeTabItemProps): ReactElement {
    return <>{children}</>;
}

interface CodeTabsProps {
    children: ReactNode;
}

export function CodeTabs({ children }: CodeTabsProps): ReactElement {
    const items = Children.toArray(children).filter(
        (c): c is ReactElement<CodeTabItemProps> => isValidElement(c),
    );
    if (items.length === 0) {
        return <></>;
    }
    const [active, setActive] = useState(items[0].props.value);
    const activeItem =
        items.find((it) => it.props.value === active) ?? items[0];
    return (
        <div className={styles.codeContainer}>
            <div className={styles.codeTabs}>
                {items.map((it) => (
                    <button
                        key={it.props.value}
                        type="button"
                        className={clsx(
                            styles.codeTab,
                            active === it.props.value && styles.codeTabActive,
                        )}
                        onClick={() => setActive(it.props.value)}
                    >
                        {it.props.label}
                    </button>
                ))}
            </div>
            {activeItem}
        </div>
    );
}
