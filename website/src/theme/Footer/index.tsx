import type { ComponentProps } from 'react';
import { useLocation } from '@docusaurus/router';
import useBaseUrl from '@docusaurus/useBaseUrl';
import OriginalFooter from '@theme-original/Footer';

export default function FooterWrapper(props: ComponentProps<typeof OriginalFooter>) {
    const { pathname } = useLocation();
    const home = useBaseUrl('/');
    const homeNoSlash = home.replace(/\/$/, '');
    const isHome = pathname === home || pathname === homeNoSlash;
    if (!isHome) {
        return null;
    }
    return <OriginalFooter {...props} />;
}
