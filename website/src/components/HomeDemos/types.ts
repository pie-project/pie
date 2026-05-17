export type EventKind =
  | 'text'
  | 'burst'
  | 'stream'
  | 'info'
  | 'section'
  | 'good'
  | 'bad'
  | 'rewind'
  | 'result';

export interface DemoEvent {
  t: number;
  kind: EventKind;
  value: string;
  fork?: string;
}

export interface DemoPane {
  label: string;
  note?: string;
  tone: 'warn' | 'good';
  events: DemoEvent[];
}

export interface DemoCode {
  language: string;
  value: string;
}

export interface DemoTrace {
  id: string;
  title: string;
  tagline: string;
  question: string;
  naive: DemoPane;
  pie: DemoPane;
  code?: {
    naive: DemoCode;
    pie: DemoCode;
  };
  runCommand?: string;
}
