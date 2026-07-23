%%%%%%%% MLSys 2025 / 2026 LaTeX submission template %%%%%%%%
% MLSys 2026 uses the same submission format as 2025.
%
% REWRITE (2026-07-20): five-operation stable waist.
% The previous full draft remains in main_old.tex. Bracketed values are
% measurement placeholders and must not be replaced without artifact evidence.

\documentclass{article}

\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amssymb}
\usepackage[hypertexnames=false]{hyperref}

\newcommand{\theHalgorithm}{\arabic{algorithm}}
\usepackage{mlsys2025}
\newtheorem{proposition}{Proposition}

\mlsystitlerunning{PLEX: Programming the Request Lifecycle in LLM Serving Systems}

\begin{document}

\twocolumn[
\mlsystitle{\texorpdfstring{PLEX: Programming the Request Lifecycle\\
in LLM Serving Systems}{PLEX: Programming the Request Lifecycle in LLM Serving Systems}}

\mlsyssetsymbol{equal}{*}

\begin{mlsysauthorlist}
\mlsysauthor{Anonymous Author(s)}{equal,inst}
\end{mlsysauthorlist}

\mlsysaffiliation{inst}{Affiliation omitted for anonymous review}
\mlsyscorrespondingauthor{Anonymous Author(s)}{anon@example.com}
\mlsyskeywords{LLM Serving, Inference Engines, Agentic Workloads, Scheduling,
KV Cache, Control Plane, WebAssembly, Extensibility}

\vskip 0.3in

\begin{abstract}
LLM serving engines expose prompts and tokens but keep the policies governing
admission, placement, scheduling, and state residency behind engine-specific
boundaries. Agentic applications make that boundary costly: one unit of work
can span several generations and tool pauses, while the application knows
workflow structure that the engine cannot infer. Passing that context as
metadata is insufficient when the policy that acts on it remains fixed, and
today testing a new policy commonly requires an engine fork.

\textbf{PLEX} makes the request lifecycle programmable. We define a
\emph{logical request} as one admitted unit of serving work and derive a
five-operation policy waist: \texttt{admit}, \texttt{route},
\texttt{schedule}, \texttt{evict}, and \texttt{feedback}. The first four
control lifecycle entry and the three recurring resource arbitrations; the
last closes the loop with enacted outcomes. Policies receive typed
host-observed facts and untrusted request metadata, and share state through
typed maps. Thin adapters retain engine mechanics and feasible-action
enforcement, while a WebAssembly host verifies, meters, and atomically replaces
operator-installed policy packages. The fourteen trigger/resource
combinations observed in a corpus of {[$K$]} prior artifacts collapse into the
five core operations and two explicitly optional operations without
{[measured loss of coverage]}. Our prototype reproduces five serving-policy
forks in 20--89 lines each, runs unchanged across {[$N$]} engines, and adds
{[measured overhead]}, while enabling coordinated policies that improve
{[workflow metric]} by {[$X$\%]} over independently loaded policies.
\end{abstract}
]

\printAffiliationsAndNotice{\mlsysEqualContribution}

\setlength{\footskip}{30pt}
\cfoot{\thepage}
\fancypagestyle{mlsysfirstpage}{%
  \fancyhf{}%
  \cfoot{\thepage}%
  \renewcommand{\headrulewidth}{0pt}%
}
\thispagestyle{mlsysfirstpage}

\begin{center}
\emph{Draft --- for internal circulation. Author list and affiliations omitted
for anonymous review.}
\end{center}

\section{Introduction}
\label{sec:intro}

An LLM serving engine and the applications above it communicate through a
narrow interface: an application submits a prompt and sampling parameters,
and the engine streams tokens. Behind that interface the engine decides which
request runs next, which KV blocks remain resident, whether to preempt, and
where to route. Those decisions are policies---FCFS scheduling, LRU eviction,
recompute on preemption---fused to the mechanics that enact them.

That arrangement fit a request lifecycle that was short and self-contained.
Agentic applications change its shape. An agent invokes the model, pauses for
a tool, continues with the result, and may repeat the loop several times. It
knows that a paused invocation will return, which future step will reuse a
prefix, and which calls belong to one user-facing task. An engine sees
generations and token sequences; it cannot reconstruct all of this program
structure. Yet that structure determines whether preserving KV avoids a long
prefill, whether locality should outweigh queue length, and whether fairness
should accumulate across calls.

The first wave of agent-aware serving systems demonstrates the value of such
information: program-level fairness, workflow-aware eviction, tool-call
continuation, cache-locality routing, and task-level accounting report
substantial gains~\cite{sheng2024fairness,abhyankar2024infercept,
luo2025autellix,fu2024efficient}. Almost every result, however, is delivered
as a fork of one engine at one version. The policy hypothesis is tens of
lines; the surrounding surgery is thousands, must be repeated across engines,
and decays as internals change.

The obvious clean alternatives miss one side of the boundary. Request
metadata carries knowledge but leaves a fixed engine policy in charge.
Router plugins can choose a backend but cannot schedule its batch or reclaim
its KV. In-process engine plugins reach those decisions but run unsandboxed
against unstable internals. The application knows the workload but does not
own the resources; the engine owns the resources but does not know the
workload. \emph{What has to cross the engine boundary is not context alone but
the policy that acts on it.}

PLEX is the narrow contract for that crossing. We first name the policy's
persistent subject: a \emph{logical request} is one admitted unit of serving
work, possibly realized as several generations linked by trusted
continuations. We then separate control from mechanics and identify three
recurring resource arbitrations: where work runs (placement), who runs and for
how long (service), and what state remains resident (residency). These yield
three policy operations---\texttt{route}, \texttt{schedule}, and
\texttt{evict}---bracketed by \texttt{admit} at lifecycle entry and
\texttt{feedback} after execution. Two mechanics-heavy operations,
\texttt{prefetch} and \texttt{rebalance}, are explicit optional extensions.

PLEX policies are operator-installed WebAssembly components. A hook receives
typed host facts and typed, untrusted request metadata. Durable configuration,
learned state, and coordination use one typed-map abstraction. The host
resolves symbolic fields and maps to compact handles at attachment, verifies
required capabilities, supplies only feasible candidates, and stages map
writes until a hook succeeds. Adapters translate each engine's internals into
the contract but leave batch construction, allocation, transfer, and kernels
inside the engine.

The design follows a lesson from extensible systems: stabilize kinds of
authority, not every event that may ever occur. A continuation and a fresh
generation may both require placement, so both invoke \texttt{route}; a
boundary and a preemption are facts, so both enter \texttt{feedback}. New
signals and events link additively without multiplying program types. The
fourteen trigger/resource combinations we found in [$K$] prior artifacts
therefore remain useful corpus coordinates, but they are evidence for the
five-operation waist rather than fourteen ABI callbacks.

This paper makes four contributions:

\begin{itemize}
\item \textbf{A programmable subject and stable waist.} We define the logical
  request and derive five core operations from lifecycle entry, three resource
  arbitrations, and closed-loop feedback
  (\S\ref{sec:waist}).
\item \textbf{A small extensible contract.} Policies use typed fields and maps;
  attachment links symbolic extensions, checks capabilities, and preserves
  provenance without exposing engine mechanics
  (\S\ref{sec:programming}).
\item \textbf{A portable implementation.} A generic Wasm host and thin
  adapters provide bounded execution, transactional state, replacement, and
  engine-default fallback (\S\ref{sec:system}).
\item \textbf{Evidence from policies and engines.} We map the first-wave
  corpus to the new surface, reproduce five forks in 20--89 lines, compare
  identical binaries across [$N$] engines, and measure composition,
  extensibility, and overhead (\S\ref{sec:eval}).
\end{itemize}

\begin{figure}[t]
\centering
\framebox[\columnwidth]{\parbox[c][5.0cm][c]{0.92\columnwidth}{\centering
\emph{Placeholder --- Figure 1.}\\[2pt]
\footnotesize Left: one engine generation, from prompt to stop. Right: one
logical request spanning generations $G_0,G_1,G_2$, with tool pauses,
single-use continuation capabilities, persistent accounting, and KV state
that may outlive each generation. Optional workflow edges relate this logical
request to siblings without merging their identities.}}
\caption{The policy subject has outgrown one generation. A logical request is
admitted once, may pause and continue across generations, and reaches one
terminal outcome.}
\label{fig:lifecycle}
\end{figure}

\section{Why Policy Must Cross}
\label{sec:motivation}

\subsection{Context without authority}
\label{sec:no-winner}

Agentic traffic creates several recurring information advantages.
Tool-calling loops know whether a boundary is a pause and how long the tool is
expected to run. Pipelines know which later stage will reuse a prefix.
Fan-out programs know the task or bundle over which fairness and deadlines
should be accounted. These are approximations to the same future knowledge
that makes Belady's eviction and shortest-remaining-time scheduling powerful
\cite{belady1966study}.

Sending these facts as request metadata is necessary but not sufficient.
Someone must decide how a hint competes with queueing delay, memory pressure,
and other tenants. InferCept, for example, observes that no single handling
strategy for intercepted requests wins across workloads
\cite{abhyankar2024infercept}. If the engine hardwires the interpretation of
each new field, extensibility remains an engine release process, and one fixed
interpretation must serve applications with different objectives.

Control from above has the opposite limitation. A gateway can reorder
arrivals and select replicas, but scheduling and eviction arbitrate over
engine-private queues, batches, and resident units on the engine's timescale.
The application has context without authority; the engine has authority
without context.

\subsection{Extension seams stop at system boundaries}

The serving stack is becoming more extensible, but along system-specific
seams. llm-d and the Kubernetes Gateway API Inference Extension compose
endpoint selection from filters and scorers~\cite{llmd,gie}; SMG exposes
WebAssembly middleware around requests~\cite{smg}; vLLM loads out-of-tree
plugins and KV connectors~\cite{vllm_plugins,vllm_kvconnector}. These efforts
confirm the demand but not a common policy contract. Router extensions cannot
reach engine scheduling or residency. Engine plugins inherit engine classes,
execute in-process, and track one release line. Per-request programmable
systems such as Pie and AICI safely control one request's own generation, but
cross-request arbitration is not one tenant's authority
\cite{gim2025pie,aici}.

The missing unit of extensibility is therefore neither an HTTP middleware
callback nor arbitrary engine code. It is a system policy resolving a typed
resource decision while the host retains feasibility and execution.

\subsection{One hypothesis, one fork}
\label{sec:forks}

Without that seam, prior systems create one. Table~\ref{tab:autopsy}
summarizes five examples. Each hypothesis is a small ranking or accounting
rule, yet it is delivered through a much larger patch against one engine
version.

\begin{table*}[t]
\centering
\caption{Anatomy of five agent-aware policy forks. Bracketed values await the
artifact audit. ``Historical point'' is the corpus label, not a PLEX ABI
entry.}
\label{tab:autopsy}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}l l r r p{0.19\textwidth} p{0.18\textwidth} c@{}}
\toprule
\textbf{Hypothesis} & \textbf{Pinned engine} & \textbf{Diff LOC} &
\textbf{Policy LOC} & \textbf{Historical point(s)} &
\textbf{Required input} & \textbf{Applies today?} \\
\midrule
Program fairness & vLLM [v0.x] & [$\sim$x{,}xxx] & [32] &
\texttt{scheduling.step} & program ID, attained service & [No] \\
Workflow eviction & vLLM [v0.x] & [$\sim$x{,}xxx] & [30] &
\texttt{caching.pressure} & steps to execution & [No] \\
Tool continuation & SGLang [vx.x] & [$\sim$x{,}xxx] & [24] &
\shortstack[l]{\texttt{caching.boundary}\\
\texttt{scheduling.}\\\texttt{boundary}} &
stop reason, resume hint & [No] \\
Locality routing & [router, vx.x] & [$\sim$xxx] & [28] &
\texttt{routing.arrive} & workflow ID, cache reuse & [Partial] \\
Bundle fairness & vLLM [v0.x] & [$\sim$x{,}xxx] & [55] &
\texttt{scheduling.step} & bundle ID & [No] \\
\bottomrule
\end{tabular}
\end{table*}

Two costs recur. The patch-to-policy ratio is roughly two orders of magnitude,
because most code creates an attachment seam rather than expressing the
hypothesis. And the cost repeats: scheduler and cache rewrites invalidate the
patch even when the policy is unchanged. Forked hypotheses are also hard to
compare or compose because they target different engines and overlapping
internals.

These failures define the requirements for PLEX. A useful crossing must grant
authority over real resource decisions, retain state across generations,
isolate untrusted policy code, abstract engine internals, and evolve without
adding one ABI entry per new workload event.

\section{The Stable Waist of the Request Lifecycle}
\label{sec:waist}
\label{sec:model}
\label{sec:derivation}

\subsection{The subject: logical requests}
\label{sec:subject}

An engine's natural work unit is a \emph{generation}: a prompt arrives,
tokens stream, and the sequence stops. Policy often needs a larger but still
bounded subject. We call a \emph{logical request} one admitted unit of serving
work, possibly realized as a sequence of generations linked by trusted
continuations, and ending in exactly one terminal outcome: completion,
cancellation, or expiry.

Identity is established by capability rather than caller declaration. At a
pausing boundary, the host may issue a single-use, expiring continuation
capability. Presenting it links the follow-up generation to the existing
logical request. A caller cannot forge continuity across principals, and
expiry or cancellation terminates the logical request. A standalone call is
simply a one-generation logical request.

The distinction is behavioral, not an engine implementation requirement.
Some engines keep one internal request open during a tool call; others end it
and later create another. The adapter binds either representation to the same
host identity. Likewise, preemption and restart remain in one generation,
while prefill-to-decode handoff changes execution stage but not identity.

What persists is the reason to name the subject. While blocked, a logical
request occupies no running batch slot, yet it leaves a \emph{physical
shadow}---KV or other state resident in the fleet---and an \emph{accounting
shadow}---arrival time, attained service, SLO debt, metadata, and placement
history. Retention, locality, and fairness decisions act on this continuity
\cite{abhyankar2024infercept,sheng2024fairness,luo2025autellix}.

Logical request does not mean ``candidate everywhere.'' Scheduling ranks
runnable generations; eviction ranks resident units, including shared units
that may benefit several requests. Logical identity is the durable
attribution and accounting key connecting those objects.

\subsection{Control, mechanics, and three arbitrations}
\label{sec:machine}
\label{sec:axes}

Everything an engine does divides into choice and execution. Choosing a
resident unit to reclaim is control; freeing or copying its bytes is
mechanics. Choosing which generations receive service and token budgets is
control; constructing a tensor batch and launching a kernel is mechanics.
Choosing a target is control; transferring KV is mechanics. PLEX programs
control and leaves mechanics to engines.

Model the fleet as a nondeterministic machine
$M=(S,\Sigma,\rightarrow)$. A state includes queues, running generations,
resident units, replica capacity, and the logical-request table. A transition
is labeled by a feasible control action followed by engine mechanics. An
execution is a serving schedule, and a policy resolves $M$'s nondeterminism at
states with more than one feasible successor. This framing has two useful
consequences. First, the host---not policy code---defines the feasible set and
enforces memory and progress bounds. Second, an adapter is behaviorally
correct when the engine's observed decisions are consistent with the loaded
policy over the candidates it exposed.

The recurring choices distinguish three types of object:

\begin{itemize}
\item \textbf{Placement:} targets competing to serve a generation or stage.
\item \textbf{Service:} waiting and running generations competing for the next
  service opportunity and token budget.
\item \textbf{Residency:} independently reclaimable units competing for
  memory, such as KV blocks, radix leaves, LoRA weights, or encoder outputs.
\end{itemize}

These are arbitration subjects, not an ontology of all serving resources.
Adding a memory tier changes residency candidates; disaggregating prefill and
decode changes placement candidates; changing continuous batching changes
mechanics. A genuinely new operation is justified only if a policy must rank
a new kind of subject with a new kind of authority.

\subsection{Five core operations}
\label{sec:hooks}
\label{sec:grid}

The stable waist is three resource decisions bracketed by lifecycle entry and
closed-loop outcome. Table~\ref{tab:hooks} gives the complete core.

\begin{table*}[t]
\centering
\caption{PLEX's policy surface. Core operations name stable kinds of authority;
causes and events explain why they run. Auxiliary operations require mechanics
that are not portable across all engines.}
\label{tab:hooks}
\footnotesize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}l l p{0.20\textwidth} p{0.25\textwidth} p{0.23\textwidth}@{}}
\toprule
\textbf{Operation} & \textbf{Class} & \textbf{Direct subject} &
\textbf{Example invocation} & \textbf{Policy result or effect} \\
\midrule
\texttt{admit} & core & new logical request & initial submission &
accept, defer, or reject \\
\texttt{route} & core & feasible placements & generation arrival; P/D stage
transition & dense placement scores \\
\texttt{schedule} & core & runnable generations & enqueue or engine service
step & dense service scores and token budgets \\
\texttt{evict} & core & resident units & allocation deficit or memory
watermark & dense retention scores \\
\texttt{feedback} & core & enacted outcome records & committed progress;
boundary; preemption; terminal outcome & staged typed-map updates \\
\midrule
\texttt{prefetch} & auxiliary & loadable state & anticipated continuation or
reuse & dense prefetch value \\
\texttt{rebalance} & auxiliary & migration options & drain, imbalance, or
hotspot & dense migration value \\
\bottomrule
\end{tabular}
\end{table*}

\paragraph{Admission.}
\texttt{admit} is called once for a new logical request. A continuation is
not admitted again. The host owns deferred submissions and defines a bounded
reconsideration rule.

\paragraph{Placement.}
\texttt{route} is one operation regardless of cause. At generation arrival it
ranks replicas, pools, or an early-bound multi-stage plan. A late-binding P/D
system invokes the same operation after prefill with decode placements; a
static pipeline does not. A continuation is represented by facts such as KV
residency and elapsed pause, not a separate \texttt{resume} function.

\paragraph{Service.}
\texttt{schedule} ranks runnable generations and may assign token budgets.
The engine forms the batch. A changed service set may cause the host to
preempt, but omission from one step is not itself preemption; only an enacted
revocation later appears as feedback.

\paragraph{Residency.}
\texttt{evict} ranks only units the adapter can legally reclaim. The host
frees low-retention candidates until the deficit is met. A radix cache may
expose leaves while a paged cache exposes pages; the policy contract fixes
behavioral fields, not engine data structures.

\paragraph{Feedback.}
\texttt{feedback} receives batched facts about what actually happened:
committed prompt and output tokens, service time, boundaries, continuations,
preemptions, and terminal outcomes. It updates learned or host-consumed maps
for later calls. It is step-batched rather than token-callback based, and raw
tokens, logits, and log-probabilities are outside the core.

This separation prevents trigger proliferation. Tool calls added new boundary
and continuation events, not new kinds of placement or service authority.
Disaggregation added a placement cause, not a new placement API. A future
retry or fork event extends the lifecycle vocabulary and invokes an existing
operation unless it creates a genuinely different arbitration.

\subsection{Auxiliary operations}

\texttt{prefetch} chooses state to load within a byte or bandwidth budget.
The adapter still resolves source, transfer, allocation, and failure.
\texttt{rebalance} ranks live migration options under a migration budget.
It is meaningful only where the deployment can transfer scheduler and KV
state, reserve destination capacity, and recover from failure
\cite{sun2024llumnix}. Engines lacking those mechanics expose neither
operation. Explicitly separating this tier prevents a portability claim from
silently depending on emulation.

\subsection{Validation against the first wave}
\label{sec:elbow}

The earlier design represented the corpus as fourteen populated combinations
of resource plane and trigger. That grid remains useful for labeling where a
fork changed an engine, but it is too fine-grained as an ABI: arrival, resume,
and handoff all repeated placement authority, while boundaries and
preemptions often reported facts rather than making immediate choices.

We remap each historical point to the new surface. Arrival placement and P/D
handoff become causes of \texttt{route}; resume becomes request state;
scheduling arrival becomes \texttt{admit}; step and pressure arbitration
become \texttt{schedule}; cache pressure becomes \texttt{evict}; progress,
boundary, resume, preemption, and finish become \texttt{feedback}. Active
cache loading maps to optional \texttt{prefetch}, and routing under sustained
imbalance maps to optional \texttt{rebalance}. Idle maintenance remains a
host mechanism. The artifact contains the full fourteen-row crosswalk.

Across [$K$] artifacts, the historical labels cover [96\%] of observed policy
changes, and the remapping preserves [all/$\cdot$\%] of those policies using
five core and two auxiliary operations. Residual artifacts either modify
mechanics such as kernels and parallelism or identify [describe any unmatched
subject]. As a temporal check, a surface frozen on artifacts before [date]
covers [$\cdot$\%] of [$M$] later artifacts. These measurements test both
directions of the factoring: too few operations lose policies; too many merely
rename triggers.

\begin{figure}[t]
\centering
\framebox[\columnwidth]{\parbox[c][4.5cm][c]{0.92\columnwidth}{\centering
\emph{Placeholder --- Figure 2.}\\[2pt]
\footnotesize A logical request enters through \texttt{admit}; placement,
service, and residency form the three middle control loops; enacted outcomes
flow through \texttt{feedback} into typed maps read by later decisions.
\texttt{prefetch} and \texttt{rebalance} appear outside the core ring.}}
\caption{The five-operation stable waist: three resource decisions bracketed
by admission and feedback. Triggers select an operation; they do not create
new program types.}
\label{fig:waist}
\end{figure}

\section{Programming PLEX}
\label{sec:programming}
\label{sec:contract}
\label{sec:design}

\subsection{Policy package and attachment}
\label{sec:policies}

A PLEX policy is an operator-installed Wasm component plus a manifest. The
component may implement any subset of the five core operations; the host
default owns an unimplemented operation. A package can implement several
operations and share typed maps across them. At most one package owns a given
operation in a deployment; automatically composing two independently written
rankers is outside the contract.

Authority is split among four actors. The \emph{application} supplies typed
but untrusted request metadata. The \emph{policy author} declares hook,
metadata, map, and capability requirements. The \emph{operator} attaches the
package to principals or workloads and binds external maps. The
\emph{adapter and host} authenticate principals, supply authoritative facts
and feasible candidates, validate results, and enact them through engine
mechanics.

Attachment is the approval and linking boundary. Installing a package
normally approves the metadata schema it declares; administrators need not
maintain a second per-field allowlist. They may still restrict tenants,
namespaces, aggregate metadata bytes, or map capacity. External
maps are bound at attachment, so a request can provide a lookup key but cannot
choose an arbitrary table for policy code to read.

\subsection{Typed operations}
\label{sec:invocation}

The conceptual SDK presents ordinary typed functions:

\begin{figure*}[t]
\begin{small}
\begin{verbatim}
admit(request, ctx) -> Accept | Defer | Reject

route(cause, request, placements, ctx) -> [Score]

schedule(runnable, capacity, ctx)
    -> [ServiceDecision { score, token_budget }]

evict(pressure, resident_units, ctx) -> [RetentionScore]

feedback(records, mutable_ctx) -> ()
\end{verbatim}
\end{small}
\caption{The five core signatures. Candidate-returning operations use dense
arrays aligned with their inputs.}
\label{fig:signatures}
\end{figure*}

Dense outputs avoid candidate identifiers crossing the wire twice. Their
length must equal the candidate count; a wrong length, NaN, trap, or deadline
miss invalidates the call and selects the engine default. Ties retain input
order. The adapter supplies only feasible candidates, and the host applies
scores through a fixed fill rule: descending score within capacity. For
\texttt{schedule}, a zero budget means no service in that opportunity, while
host-enforced progress floors prevent permanent starvation. Whether an
adapter can enact per-request token budgets is a required capability rather
than a silent approximation.

Candidate-local and set-dependent policies are distinct. A candidate-local
score can be refreshed on enqueue and consumed by a native priority index.
A set-dependent policy must see the batch at each selection point. Similarly,
an eviction policy may maintain per-unit values or rank a full pressure set.
The package declares its dependence mode; the host cannot change invocation
mode as a transparent performance optimization.

\subsection{Fields and maps}
\label{sec:signals}

PLEX has two data mechanisms. \emph{Invocation fields} travel with hook
inputs. \texttt{facts()} exposes host-observed values such as authenticated
principal, queue depth, attained service, and KV residency.
\texttt{metadata()} exposes application-declared values such as workflow ID,
expected output length, and preferred region. The accessor preserves
provenance and absence:

\begin{verbatim}
let served = r.facts()
              .attained_service();
let hint = r.metadata()
            .get(EXPECTED_TOKENS);
\end{verbatim}

The policy package declares metadata names, scalar types, scope
(logical-request or generation), and size limits. A request may then carry:

\begin{verbatim}
"plex": { "metadata": {
  "acme.workflow-id@1": "wf-123",
  "acme.expected-output-tokens@1": 192
}}
\end{verbatim}

The gateway validates this object against the attached schema and materializes
typed fields. Request-scoped values persist across trusted continuations;
generation-scoped values do not. Host identities and accounting fields occupy
reserved namespaces and cannot be overwritten. Well-typed metadata remains a
claim by the caller, not an observed fact.

\emph{Typed maps} hold all durable keyed data. The same API covers external
operator or tenant configuration, policy-owned learned state, and bounded
host-consumed intent:

\begin{verbatim}
ctx.map(TENANT_CONFIG)
   .get(principal);
ctx.map(ACCOUNTING)
   .add(request_id, delta);
ctx.map(RETENTION)
   .upsert(request_id, intent);
\end{verbatim}

A map declaration fixes key and value schemas, writer, access, capacity, and
persistence. External maps are policy read-only. Policy maps stage
	exttt{upsert}, \texttt{add}, and \texttt{delete} operations. Host-consumed
maps use standard schemas for effects such as a
bounded service reservation or retention intent; their values carry TTLs and
count against per-policy quotas. This retains one map mechanism without
letting an arbitrary private map acquire side effects by naming convention.

The following abbreviated scheduler uses all four provenance classes:

\begin{figure}[t]
\begin{small}
\begin{verbatim}
fn schedule(cands, cap, ctx) {
  cands.map(|r| {
    let served = r.facts()
                  .attained_service();
    let hint = r.metadata()
                .get(EXPECTED_TOKENS);
    let cfg = ctx.map(TENANT_CONFIG)
                 .get(
                   r.facts().principal())
                 .unwrap_or(DEFAULT);
    let id = r.facts().logical_id();
    let state = ctx.map(ACCOUNTING)
                   .get(id)
                   .unwrap_or_default();
    Service {
      score: cfg.weight /
             (1.0 + served + state.debt),
      budget: state.predicted.or(hint)
              .unwrap_or(cap.default),
    }
  })
}
\end{verbatim}
\end{small}
\caption{A policy reads host facts, request metadata, operator configuration,
and policy-owned state without losing provenance.}
\label{lst:policy}
\end{figure}

Map writes are transactional with the hook. The SDK appears imperative, but
the host stages operations while recording the revisions read by the policy.
For a decision hook, the host validates and prepares the transaction, the
adapter revalidates and enacts the decision, and only successful enactment
commits the effects. A conflict, failed decision, or failed enactment discards
them. Outcome-dependent updates for operations without synchronous enactment
acknowledgement must arrive through \texttt{feedback}. A feedback delivery ID,
its map updates, and acknowledgement commit together, so replay cannot double
count service within the durability scope of the map backend.

\begin{proposition}[Observed substitution is strategyproof]
\label{prop:sp}
Consider a scheduler that orders logical requests only by cumulative
host-observed attained service and serves the least-served first. If tenants
can choose metadata but cannot alter observed service, allocation is
independent of every declaration. Truthful declaration or omission is
therefore weakly dominant.
\end{proposition}

Provenance does not impose this policy; it makes such defenses expressible.
A mixed policy may compare predicted with realized length in
\texttt{feedback}, store credibility in a principal-scoped map, and discount
repeated misdeclaration.

\subsection{An ABI that can grow}
\label{sec:portability}

The conceptual SDK is typed, but the wire representation cannot assume that
WIT records and variants grow in place: adding a case or field creates a new
component type. PLEX therefore distinguishes closed actions from open
vocabularies. Host-interpreted actions such as \texttt{Admission} remain
closed WIT variants. Symbolic metadata, event, capability, and map names in
the manifest resolve to compact local handles at attachment. Stable hot-path
fields live in versioned core records; long-tail values use homogeneous typed
columns selected by those handles.

Required and optional dependencies have only two outcomes. A missing required
hook capability, event, field, or map rejects attachment. A missing optional
item is absent or not delivered. PLEX does not silently coerce an action to a
``nearby'' mechanic, because that makes behavioral portability impossible to
measure.

Linking avoids one host call per field: the host materializes required facts
and metadata into candidate columns before crossing the sandbox. Maps remain
host-backed, and SDK accessors may look up keys computed during the invocation
through attachment-linked handles. Calls and returned bytes are metered and
bounded. A manifest may request pre-joined values for common key-source fields
as a hot-path optimization, but this does not change lookup semantics. Policy
code neither parses JSON nor performs network I/O in a hook.

\subsection{One logical request end to end}
\label{sec:worked}

Consider a tool-calling logical request in a ten-step workflow. Its first
generation arrives with a workflow ID and expected output length as metadata.
\texttt{admit} accepts it against host load and operator limits.
\texttt{route(Arrival)} combines a preferred-region hint with observed prefix
overlap and queueing, then selects a feasible replica. \texttt{schedule}
combines the tenant's operator weight, logical-request attained service, and
learned length state to assign service and a token budget.

After each engine step, one batched \texttt{feedback(Progress)} adds committed
tokens and service time to the accounting map. The model emits a tool call.
\texttt{feedback(Boundary)} records a bounded retention intent keyed by the
logical request; if the policy reserves future service, it writes a standard
service-reservation map with a host-enforced TTL. While the tool runs,
\texttt{evict} sees resident units annotated with enacted host facts and
retention value and reclaims lower-value state.

The follow-up presents the continuation capability, creating a new generation
under the same logical request. It is not admitted again.
\texttt{route(Arrival)} now sees where its KV remains resident, and
\texttt{schedule} sees accumulated service rather than resetting fairness.
On completion, \texttt{feedback(Finished)} performs terminal accounting and
removes outstanding intent entries. The hooks coordinate through typed maps,
but every physical action remains an engine transition.

\section{The System}
\label{sec:system}

PLEX consists of a policy package, a generic host, and thin router and engine
adapters (Figure~\ref{fig:arch}).

\begin{figure}[t]
\centering
\framebox[\columnwidth]{\parbox[c][4.8cm][c]{0.92\columnwidth}{\centering
\emph{Placeholder --- Figure 3.}\\[2pt]
\footnotesize Application metadata and operator/tenant maps enter a PLEX host.
At attachment the host links schemas and capabilities to a Wasm policy
package. Router and engine adapters provide facts and feasible candidates;
typed dense decisions return to native mechanics. A staged map store connects
hooks and survives replacement when pinned.}}
\caption{PLEX architecture. The contract carries policy and typed data;
adapters retain engine-specific mechanics.}
\label{fig:arch}
\end{figure}

\subsection{Host and hot path}

The host instantiates a Wasm component with only PLEX imports: typed map
operations, a monotonic clock, bounded entropy, and telemetry. It exposes no
filesystem, socket, arbitrary memory, or engine object. At attachment it
validates the manifest, binds external maps, creates policy maps, resolves
symbolic names, and rejects unsatisfied required capabilities.

For each invocation, the adapter produces core candidate records and typed
extension columns. The host joins declared metadata and bound-map values,
charges the resulting bytes against policy quotas, and enters Wasm once with
the batch. Dense results are validated before the adapter sees them. A
candidate-local enqueue or standing-index mode avoids a sandbox crossing on
every scheduler or allocator action; set-dependent policies opt into batched
synchronous calls and their measured cost.

\subsection{Isolation, availability, and policy quality}
\label{sec:safety}

Three guarantees must not be conflated. \emph{Mechanical isolation} comes
from Wasm memory isolation and a typed host surface: policy code cannot mutate
engine objects or request contents. \emph{Availability} comes from deadlines,
bounded allocation, map quotas, and engine-default fallback. A
trap, timeout, malformed dense result, or invalid map update discards the
transaction and applies the native heuristic for that invocation.

\emph{Policy quality} is not guaranteed by sandboxing. A valid ranker may make
poor decisions or favor one tenant. The host bounds effects it owns:
deferral is capped, host-consumed intents expire and consume quotas, and a
progress watchdog may detach a package whose workloads stop advancing.
Beyond those bounds, operators remain responsible for the policy they attach.

Telemetry follows a separate lossy path. Policies may emit bounded metrics or
debug records to a ring buffer, but no decision or map commit depends on
telemetry delivery.

\subsection{Transactional maps and replacement}

Each invocation receives immutable facts and a stable map snapshot. Host-backed
reads record entry revisions, and policy-map updates remain staged. Prepare
checks that every observed revision is current and briefly fences the write set
through native enactment. A conflict aborts the decision and its effects; the
adapter may retry from a fresh snapshot within the operation deadline and must
otherwise invoke its native fallback. Successful enactment, feedback
deduplication where applicable, and prepared writes then commit together. This
is process-local atomicity, not crash-atomic coordination with external engine
state. External configuration is published as immutable, monotonically
increasing revisions so one candidate batch never mixes versions.

Attachment creates a link between one package export and each operation it
owns. Replacing a link is atomic: new calls enter the new component while
in-flight calls finish under the old one. Attachment-scoped maps disappear
with the package; pinned maps transfer only when schemas match. A failed
activation rolls back to the prior link, and detach always restores the engine
default.

\subsection{Adapters and attachability}
\label{sec:impl}

Adapters translate semantic views rather than expose engine structures.
Routing usually attaches at a cluster router. Admission and scheduling attach
where an engine accepts and selects work. Eviction attaches at the engine's
legal reclaim unit. Feedback taps committed execution and lifecycle changes.
An adapter manifest records native and amortized invocation modes, deadlines,
candidate semantics, fields, and auxiliary mechanics.

Table~\ref{tab:matrix} is the evaluation target; bracketed entries require
prototype confirmation. ``Absent'' is preferable to an emulation that changes
policy meaning.

\begin{table*}[t]
\centering
\caption{Operation attachability by target. N = native, A = amortized/indexed,
E = explicit emulation with measured semantics, -- = absent. Values remain to
be confirmed by prototypes.}
\label{tab:matrix}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}l c c c c c c c r@{}}
\toprule
\textbf{Target} & \textbf{route} & \textbf{admit} & \textbf{schedule} &
\textbf{evict} & \textbf{feedback} & \textbf{prefetch} &
\textbf{rebalance} & \textbf{Adapter LOC} \\
\midrule
vLLM & router & [N] & [N/A] & [N/A] & [N] & [N] & [--] & [418] \\
SGLang & router & [N] & [N/A] & [N/A] & [N] & [N] & [--] & [$\sim$250] \\
TensorRT-LLM & router & [N] & [N/A] & [N/A] & [N] & [--] & [--] & [$\sim$xxx] \\
Dynamo & [N] & [N] & [N/A] & [--] & [N] & [N] & [N] & [256] \\
Pie & router & [N] & [N] & [N] & [N] & [N] & [--] & [$\sim$xxx] \\
\bottomrule
\end{tabular}
\end{table*}

The current artifact contains a generic host of [$\sim$1{,}373] Rust lines, a
PyO3 path of [$\sim$296] lines for Python engines, [32] reference policies,
the labeled derivation corpus, and a replay harness. The harness drives
recorded candidate batches through any adapter and compares policy decisions,
fallbacks, and map transitions.

\section{Evaluation}
\label{sec:eval}

We evaluate whether the stable waist captures real policies, transfers across
engines, grows without ABI proliferation, and is practical on serving hot
paths. All bracketed quantities below are pending the corresponding artifact
run.

\paragraph{Setup.}
[Models, GPUs, engines and versions, trace sources, policy packages, map
configuration, and workload-generation parameters.]

\subsection{E1: Expressiveness}
\label{sec:eval-expressiveness}
\label{sec:eval-q3}

\paragraph{Corpus remapping.}
We first apply the fixed fourteen-to-5+2 protocol of
\S\ref{sec:elbow} to all [$K$] artifacts. Report (i) historical-grid
coverage, (ii) the fraction preserved by the five core operations, (iii) the
increment from the two auxiliary operations, and (iv) residuals that change
mechanics. A merge/split sensitivity analysis tests whether another operation
boundary yields the same coverage with less authority.

\paragraph{Reproducing fork hypotheses.}
We reimplement the five policies in Table~\ref{tab:autopsy}. Cache-aware
scheduling raises prefix-cache hit rate from [0\%] to [91\%] in [22] lines.
Bundle fairness changes the measured fairness metric from [1.65] to [0.66] in
[55] lines. Workflow-aware eviction raises workflow hit rate from [88.7\%] to
[96.0\%] in [30] lines. Program-level fairness changes light- and heavy-tenant
service by [1.11$\times$] and [1.78$\times$] in [32] lines. Tool continuation
and locality routing report [results]. For each, we compare the PLEX decision
trace and workload outcome with the source implementation, rather than
claiming parity from LOC alone.

\paragraph{Coordinated policy.}
A single [80]-line package implements routing, scheduling, eviction, and
feedback around a workflow. Routing records enacted locality in a map;
feedback records progress and bounded continuation intent; later schedule and
evict calls consume that state. We compare engine defaults, the best
single-operation policy, all operations without shared state, and the
coordinated package. Coordination improves [workflow metric] by [$X$\%] over
independent operation policies and changes [recomputed prefill/cache hit/SLO
goodput] by [$\cdot$]. The comparison isolates composition through the
contract, not merely the sum of point policies.

\subsection{E2: Portability}
\label{sec:eval-portability}
\label{sec:eval-q4}

We distinguish three claims.

\paragraph{Load portability.}
The identical component binary attaches to every target satisfying its
manifest. We report successful attachment by operation, missing required
capabilities, and optional absences using Table~\ref{tab:matrix}.

\paragraph{Contract portability.}
For a recorded invocation stream, each adapter must deliver fields with the
same units, provenance, candidate meaning, and missing-value semantics.
Report field/capability gaps, default-call rate, and whether an operation is
native or amortized. We do not hide a missing capability behind coercion.

\paragraph{Behavioral portability.}
The replay harness feeds canonical batches and map revisions to each adapter
and compares dense results, chosen candidates, and committed map transitions.
Core packages produce [bit-identical/equivalent] traces on [$N$] engines;
live runs report how different engine mechanics affect workload outcomes
despite identical policy decisions. Across an engine scheduler rewrite,
updating PLEX changes [n] adapter lines and zero policy lines, versus [m]
lines or a rewrite for the corresponding fork.

\subsection{E3: Extensibility and composition}
\label{sec:eval-composition}
\label{sec:eval-q1}
\label{sec:eval-q2}

We test the design claim that vocabulary grows without multiplying program
types:

\begin{enumerate}
\item add a namespaced metadata field and run an old binary unchanged;
\item add an optional feedback event and verify that an unsubscribed package
  receives no call;
\item add a resident-unit kind through extension columns;
\item omit a required token-budget capability and verify attach-time rejection;
\item replace a package atomically while preserving a schema-compatible pinned
  accounting map; and
\item share accounting and intent maps across \texttt{feedback},
  \texttt{schedule}, and \texttt{evict}.
\end{enumerate}

For each case, report binary changes, attachment result, decision-trace
changes, replacement interruption, and state loss. If run, a temporal corpus
holdout freezes the surface before [date] and measures coverage of [$M$] later
artifacts.

We then replay [three agentic trace classes] against engine defaults and the
coordinated package. Report workflow completion latency (p50/p99), SLO
goodput, recomputed prefill tokens, and prefix-cache hit rate. The best policy
is expected to vary by trace; the policy-by-workload matrix tests the
no-universal-policy premise of \S\ref{sec:no-winner}.

\subsection{E4: Practicality}
\label{sec:eval-practicality}
\label{sec:eval-q5}

\paragraph{Hot-path cost.}
Measure candidate materialization, map pre-join, sandbox crossing, policy
execution, result validation, and commit separately. Report per-hook latency
versus candidate count for candidate-local and set-dependent modes; batched
\texttt{feedback} cost versus records per call; map read, staged add, and
commit cost; and end-to-end throughput, TTFT, and token latency. The full
package costs [$\cdot\mu$s / $\cdot$\%] per [invocation] and [$<x\%$]
throughput at [load].

\paragraph{Failure behavior.}
Inject traps, infinite loops, malformed dense arrays, NaNs, allocation
attempts, and invalid map updates. Verify default fallback, transaction
rollback, and continued progress; report detection latency and fallback rate.
Separately test poor but valid policies (permanent deferral, starvation, and
over-reservation) to measure the limits of host caps and watchdog ejection.

\paragraph{Untrusted metadata.}
One tenant inflates predicted length, urgency, and reuse hints. Compare a
naive policy, the observed-service policy of
Proposition~\ref{prop:sp}, and a feedback-trained credibility policy.
Report service share, honest-tenant p99, and requests until credibility
recovers. This evaluates provenance as a programming primitive rather than
claiming that PLEX automatically makes metadata truthful.

\paragraph{Engineering cost.}
Report host, binding, and per-adapter LOC; implementation time per operation;
policy LOC; and maintenance across one engine version change. Compare adapter
work paid once per engine against fork work paid per policy and version.

\section{Discussion and Limitations}
\label{sec:discussion}

\paragraph{Mechanics remain out of scope.}
\label{sec:residual}
PLEX controls choices, not attention kernels, parallelism layouts, batch
construction, or transfer implementations. The [4\%] of corpus artifacts that
change such mechanics remain engine work. This boundary is deliberate but
limits the paper's reach.

\paragraph{Core does not mean universally native.}
The five operations define the stable policy language. A deployment may
implement a hot operation through a standing index, and a target without real
placement choice may expose a singleton. Auxiliary prefetch and rebalance have
no portability promise. The evaluation must distinguish native, amortized,
explicitly emulated, and absent support.

\paragraph{A new arbitration can break the waist.}
New events, signals, stages, and resident-unit kinds should fit existing
operations. If a serving system must rank a new kind of subject with authority
not expressible as admission, placement, service, residency, or feedback, PLEX
needs a new operation. The corpus supports the current boundary; it does not
prove an ontology for all future systems.

\paragraph{Composition is explicit.}
Maps let one package coordinate hooks, but PLEX does not synthesize a coherent
policy from independently authored rankers or resolve two owners of the same
operation. One active owner per operation keeps enactment deterministic;
automatic policy composition is future work.

\paragraph{Metadata defenses are asymmetric.}
Predicted length can be compared with realized length; a claimed future
workflow edge may be observable only after it matters. Feedback can reduce
repeated abuse but cannot retroactively prevent the first misleading
declaration.

\paragraph{Contract versus implementation.}
The typed-map intent design, atomic replacement, and some auxiliary
capabilities remain subject to gates in the prototype. We mark unimplemented
features and unmeasured results rather than treating a WIT sketch as evidence.

\section{Related Work}
\label{sec:related}

\paragraph{Context-aware serving.}
Program fairness, workflow-aware eviction, tool continuation, affinity
routing, and bundle fairness motivate PLEX
\cite{sheng2024fairness,abhyankar2024infercept,luo2025autellix,
fu2024efficient}. Parrot carries application structure through semantic
variables~\cite{lin2024parrot}; Preble and MemServe build fixed prefix-aware
policies~\cite{srivatsa2025preble,hu2024memserve}. PLEX contributes neither a
new point policy nor a universal winner, but a shared authority and data
contract under which such policies can be loaded and compared.

\paragraph{Router-tier control planes.}
SMG, the SGLang gateway, llm-d, the Kubernetes Gateway API Inference
Extension, Dynamo, and AIBrix make placement increasingly configurable
\cite{smg,sglang_gateway,llmd,gie,dynamo,shan2025aibrix}. They occupy the
natural host of \texttt{route}; most cannot decide how a backend schedules or
reclaims state. PLEX includes routing but extends the same typed policy package
into engine decisions.

\paragraph{Engine and request extensibility.}
vLLM plugins and KV connectors provide valuable attachment mechanics but are
engine-specific and generally in-process
\cite{vllm_plugins,vllm_kvconnector}. Pie and AICI demonstrate sandboxed
programs for one request's generation and decoding
\cite{gim2025pie,aici}. PLEX instead grants an operator-installed policy
authority over cross-request resource arbitration and uses those systems as
potential targets rather than competitors at the same layer.

\paragraph{eBPF, \texttt{sched\_ext}, and SDN.}
PLEX borrows the extensibility discipline of eBPF and
\texttt{sched\_ext}: a small set of program types, typed contexts, maps,
load-time verification, bounded execution, attach links, atomic replacement,
and fallback to host policy
\cite{soldani2023ebpf,schedext,mccanne1993bsd}. The closest correspondence is
not ``Wasm for serving'' but the separation between a stable authority surface
and an extensible vocabulary linked at load time. PLEX differs in deriving
program types from serving arbitration, preserving declared-versus-observed
provenance, and spanning a fleet of user-space engines. SDN similarly
separates control policy from forwarding mechanics
\cite{mckeown2008openflow,casado2007ethane}; OpenFlow's fixed match vocabulary
also motivates keeping PLEX's long-tail fields open.

\paragraph{Self-declared resource metadata.}
DiffServ polices declared priority at trust boundaries, and strategyproof
scheduling designs allocation rules under which lying does not pay
\cite{diffserv,nisan2001algorithmic}. PLEX does not certify declarations. It
preserves provenance and outcome feedback so an operator can choose observed
substitution, policing, or learned credibility.

\section{Conclusion}
\label{sec:conclusion}

Agentic applications changed a serving request from one isolated generation
into a logical unit that can pause, continue, and carry physical and
accounting state across calls. The information needed to govern that unit
lives above engines, while the authority to place, schedule, and retain it
lives inside them. Metadata alone cannot bridge the gap when policy remains
fixed, and forks do not provide a stable research or deployment interface.

PLEX programs the request lifecycle through a five-operation closed loop:
admission, three resource arbitrations, and feedback. Typed fields preserve
the boundary between observed facts and caller declarations; typed maps carry
configuration, learned state, and bounded host intent; adapters retain
feasibility and mechanics. The result is intended to do for serving policy
what successful extensible substrates do elsewhere: keep the authority waist
small while allowing events, data, policies, and implementations to evolve
around it.

\bibliography{references}
\bibliographystyle{mlsys2025}

\end{document}
