import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  overviewSidebar: [
    'overview/what-is-pie',
    'overview/components',
    'overview/key-features',
    'overview/comparison',
    'overview/benchmarks',
    'overview/faq',
  ],

  guideSidebar: [
    {
      type: 'category',
      label: 'Getting started',
      collapsed: false,
      items: [
        'guide/install',
        'guide/setup',
        'guide/dev-env',
        'guide/first-inferlet',
      ],
    },
    {
      type: 'category',
      label: 'Tutorial',
      collapsed: false,
      items: [
        'guide/tutorial/build',
        'guide/tutorial/run',
        'guide/tutorial/serve',
      ],
    },
    {
      type: 'category',
      label: 'Examples',
      collapsed: true,
      items: [
        'guide/examples/overview',
        'guide/examples/chat',
        'guide/examples/samplers',
        'guide/examples/structured',
        'guide/examples/kv-cache',
        'guide/examples/speculation',
        'guide/examples/reasoning',
        'guide/examples/agents',
        'guide/examples/integration',
      ],
    },
    {
      type: 'category',
      label: 'Model',
      collapsed: false,
      items: [
        'guide/model/loading',
        'guide/model/tokenizer',
        'guide/model/multimodal',
      ],
    },
    {
      type: 'category',
      label: 'Context',
      collapsed: false,
      items: [
        'guide/context/overview',
        'guide/context/pages',
        'guide/context/sharing',
        {type: 'doc', id: 'guide/context/scheduling', className: 'sidebar-experimental'},
      ],
    },
    {
      type: 'category',
      label: 'Forward pass',
      collapsed: false,
      items: [
        'guide/forward/overview',
        'guide/forward/inputs',
        'guide/forward/media',
        'guide/forward/samplers',
        'guide/forward/constrained',
        'guide/forward/speculation',
        {type: 'doc', id: 'guide/forward/adapters', className: 'sidebar-experimental'},
      ],
    },
    {
      type: 'category',
      label: 'Generation',
      collapsed: false,
      items: [
        'guide/decoder/overview',
        'guide/decoder/generator',
        'guide/decoder/chat',
        'guide/decoder/reasoning',
        'guide/decoder/tool-calling',
      ],
    },
    {
      type: 'category',
      label: 'I/O and messaging',
      collapsed: false,
      items: [
        'guide/io/overview',
        'guide/io/session',
        'guide/io/messaging',
        'guide/io/http',
        'guide/io/filesystem',
      ],
    },
    {
      type: 'category',
      label: 'Deployment',
      collapsed: false,
      items: [
        'guide/deploy/overview',
        'guide/deploy/build-publish',
        'guide/deploy/serve',
        'guide/deploy/clients',
        'guide/deploy/profiling',
      ],
    },
  ],

  referenceSidebar: [
    {
      type: 'category',
      label: 'Inferlet',
      collapsed: false,
      items: [
        'reference/sdk-rust',
        'reference/sdk-python',
        'reference/sdk-javascript',
        'reference/manifest',
      ],
    },
    {
      type: 'category',
      label: 'Clients',
      collapsed: false,
      items: [
        'reference/client-rust',
        'reference/client-python',
        'reference/client-javascript',
      ],
    },
    {
      type: 'category',
      label: 'CLI',
      collapsed: false,
      items: [
        'reference/pie',
        {type: 'doc', id: 'reference/bakery', className: 'sidebar-experimental'},
        'reference/pie-client',
      ],
    },
    {
      type: 'category',
      label: 'Drivers',
      collapsed: false,
      items: [
        'reference/drivers/cuda',
        'reference/drivers/portable',
      ],
    },
    {
      type: 'category',
      label: 'Configuration',
      collapsed: false,
      items: [
        'reference/configuration',
      ],
    },
  ],

};

export default sidebars;
