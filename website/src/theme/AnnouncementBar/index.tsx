import type { ComponentProps } from 'react';
import { useLocation } from '@docusaurus/router';
import useBaseUrl from '@docusaurus/useBaseUrl';
import OriginalAnnouncementBar from '@theme-original/AnnouncementBar';

const HIDDEN_ROUTES = ['preview'];

export default function AnnouncementBarWrapper(
    props: ComponentProps<typeof OriginalAnnouncementBar>,
) {
    const { pathname } = useLocation();
    const base = useBaseUrl('/').replace(/\/$/, '');
    const hidden = HIDDEN_ROUTES.some((route) => {
        const target = `${base}/${route}`;
        return pathname === target || pathname.startsWith(`${target}/`);
    });
    if (hidden) {
        return null;
    }
    return <OriginalAnnouncementBar {...props} />;
}
