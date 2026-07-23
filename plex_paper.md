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
\emph{request} as one admitted unit of serving work and derive a
five-operation policy waist: \texttt{admit}, \texttt{route},
\texttt{schedule}, \texttt{cache}, and \texttt{feedback}. The first four
control lifecycle entry and the three recurring resource arbitrations; the
last closes the loop with enacted outcomes. Policies receive typed
host-observed facts and untrusted request metadata, and share state through
explicit shared, work-group, and request scopes. Thin adapters retain engine
mechanics and feasible-action
enforcement, while a WebAssembly host verifies, meters, and atomically replaces
operator-installed policy packages. The fourteen trigger/resource
combinations observed in a corpus of {[$K$]} prior artifacts collapse into the
five core operations plus explicitly negotiated engine mechanics without
{[measured loss of coverage]}. The artifact implements and classifies 31
policy kernels, exercises equivalent vLLM and SGLang adapter templates, and
records validator and runtime overhead against committed regression budgets.
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
luo2025autellix}. Almost every result, however, is delivered
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
persistent subject: a \emph{request} is one admitted unit of serving
work, possibly realized as several generations linked by trusted
continuations. We then separate control from mechanics and identify three
recurring resource arbitrations: where work runs (placement), who runs and for
how long (service), and what state remains resident (residency). These yield
three policy operations---\texttt{route}, \texttt{schedule}, and
\texttt{cache}---bracketed by \texttt{admit} at lifecycle entry and
\texttt{feedback} after execution. Prefetch, cancellation, swap, migration,
and atomic enqueue remain versioned optional engine mechanics and actions
rather than additional core operations.

PLEX policies are operator-installed WebAssembly components. A hook receives
typed structural records plus bounded JSON documents for extensible facts,
fields, and scratch. Durable policy state is explicit at the component
boundary: shared state, trusted work-group state, and the exact referenced
request set. The host verifies required mechanics and schemas, supplies only
feasible candidates, and conditionally commits sparse state replacements.
Adapters translate each engine's internals into the contract but leave batch
construction, allocation, transfer, and kernels inside the engine.

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
\item \textbf{Programmable subjects and a stable waist.} We define
  \texttt{Request} and trusted \texttt{WorkGroup}, then derive five core
  operations from lifecycle entry, three resource arbitrations, and closed-loop feedback
  (\S\ref{sec:waist}).
\item \textbf{A small extensible contract.} Policies use typed structure and
  explicit scoped state; attachment checks schemas and mechanics and preserves
  provenance without exposing engine mechanics
  (\S\ref{sec:programming}).
\item \textbf{A portable implementation.} A generic Wasm host and thin
  adapters provide bounded execution, conditional policy-state commit, replacement, and
  engine-default fallback (\S\ref{sec:system}).
\item \textbf{Evidence from policies and engines.} We map 87 papers to the
  surface, implement and independently audit 31 inspired policy adaptations,
  run typed fixtures through the host, validate a live asynchronous vLLM
  attachment, and measure validator and runtime overhead (\S\ref{sec:eval}).
\end{itemize}

\begin{figure}[t]
\centering
\framebox[\columnwidth]{\parbox[c][5.0cm][c]{0.92\columnwidth}{\centering
\emph{Placeholder --- Figure 1.}\\[2pt]
\footnotesize Left: one engine generation, from prompt to stop. Right: one
request spanning generations $G_0,G_1,G_2$, with tool pauses,
single-use continuation capabilities, persistent accounting, and KV state
that may outlive each generation. A trusted WorkGroup may relate this request
to independently routed and scheduled siblings without merging identities.}}
\caption{The policy subject has outgrown one generation. A request is
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
Workflow cache policy & vLLM [v0.x] & [$\sim$x{,}xxx] & [30] &
\texttt{caching.pressure} & steps to execution & [No] \\
Tool continuation & SGLang [vx.x] & [$\sim$x{,}xxx] & [24] &
\shortstack[l]{\texttt{caching.boundary}\\
\texttt{scheduling.}\\\texttt{boundary}} &
stop reason, resume hint & [No] \\
Locality routing & [router, vx.x] & [$\sim$xxx] & [28] &
\texttt{routing.arrive} & workflow ID, cache reuse & [Partial] \\
Critical-path scheduling & [workflow engine] & [$\sim$x{,}xxx] & [55] &
\texttt{scheduling.step} & trusted readiness, dependency depth, cache reuse & [No] \\
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

\subsection{The subject: requests}
\label{sec:subject}

An engine's natural work unit is a \emph{generation}: a prompt arrives,
tokens stream, and the sequence stops. Policy often needs a larger but still
bounded subject. We call a \emph{request} one admitted unit of serving
work, possibly realized as a sequence of generations linked by trusted
continuations, and ending in exactly one terminal outcome: completion,
cancellation, or expiry.

Identity is established by capability rather than caller declaration. At a
pausing boundary, the host may issue a single-use, expiring continuation
capability. Presenting it links the follow-up generation to the existing
request. A caller cannot forge continuity across principals, and
expiry or cancellation terminates the request. A standalone call is
simply a one-generation request.

Parallel agent calls require a second, orthogonal subject. A
\emph{WorkGroup} is a host-issued coordination scope whose principal,
lifecycle, quotas, facts, and scratch can outlive any one member request.
Membership is immutable and authenticated; copying a group identifier into
metadata does not establish membership. A work group is not itself schedulable
and does not imply a DAG runtime, co-location, atomic enqueue, or simultaneous
execution.

The distinction is behavioral, not an engine implementation requirement.
Some engines keep one internal request open during a tool call; others end it
and later create another. The adapter binds either representation to the same
host identity. Likewise, preemption and restart remain in one generation,
while prefill-to-decode handoff changes execution stage but not identity.

What persists is the reason to name the subject. While blocked, a request
occupies no running batch slot, yet it leaves a \emph{physical
shadow}---KV or other state resident in the fleet---and an \emph{accounting
shadow}---arrival time, attained service, SLO debt, metadata, and placement
history. Retention, locality, and fairness decisions act on this continuity
\cite{abhyankar2024infercept,sheng2024fairness,luo2025autellix}.

Request does not mean ``candidate everywhere.'' Scheduling selects
runnable requests; cache policy evaluates resident and prospective objects,
including shared objects that may benefit several requests or groups. Request
identity is the durable
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
resident units, replica capacity, and the request/work-group tables. A transition
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
causes and events explain why they run. Non-portable mechanics remain
versioned capabilities and actions outside the five-operation waist.}
\label{tab:hooks}
\footnotesize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}l l p{0.20\textwidth} p{0.25\textwidth} p{0.23\textwidth}@{}}
\toprule
\textbf{Operation} & \textbf{Class} & \textbf{Direct subject} &
\textbf{Example invocation} & \textbf{Policy result or effect} \\
\midrule
\texttt{admit} & core & pending requests & bounded admission opportunity &
dense accept, defer, or reject decisions \\
\texttt{route} & core & request--target feasible graph & admitted work;
retry; rebalance & dense direct edge assignments or defer \\
\texttt{schedule} & core & runnable requests & service opportunity &
non-overlapping selections and token budgets \\
\texttt{cache} & core & resident and prospective objects & insertion,
pressure, expiry, or dependency progress & dense admission plus legal reclaim
order \\
\texttt{feedback} & core & enacted outcome records & committed progress;
boundary; preemption; terminal outcome & sparse policy-state updates \\
\bottomrule
\end{tabular}
\end{table*}

\paragraph{Admission.}
\texttt{admit} receives every pending request in one bounded opportunity.
The result is dense and request-aligned; accepted requests collectively obey
the supplied count and resource limits. A continuation of the same request is
not admitted again. Deferred identities remain pending for a later opportunity
with a new opportunity ID.

\paragraph{Placement.}
\texttt{route} receives a bounded request set, targets, and a sparse feasible
edge graph. It returns a direct joint assignment, so a non-greedy matching is
expressible without a host-defined score solver. A late-binding P/D system can
invoke the same operation with stage-specific targets. A continuation is
represented by facts such as KV residency and elapsed pause, not a separate
\texttt{resume} function.

\paragraph{Service.}
\texttt{schedule} returns explicit non-overlapping selections and aligned
token budgets. A multi-request selection is all-or-none in the normalized
plan. Atomic adapter enqueue is a negotiated mechanic; simultaneous GPU start
is not a core guarantee. Omission from one step is not itself preemption; only
an enacted revocation later appears as feedback.

\paragraph{Residency.}
\texttt{cache} evaluates resident and prospective objects in one snapshot.
It can bypass a prospective object even with free capacity, admit a valuable
object while reclaiming a lower-value resident, and return an explicit legal
reclaim order. Dependency-constrained caches expose a bounded iterative
episode whose eligible frontier changes after enacted steps.

\paragraph{Feedback.}
\texttt{feedback} receives batched facts about what actually happened:
committed prompt and output tokens, service time, boundaries, continuations,
preemptions, and terminal outcomes. It updates learned or host-consumed state
for later calls. It is step-batched rather than token-callback based, and raw
tokens, logits, and log-probabilities are outside the core.

This separation prevents trigger proliferation. Tool calls added new boundary
and continuation events, not new kinds of placement or service authority.
Disaggregation added a placement cause, not a new placement API. A future
retry or fork event extends the lifecycle vocabulary and invokes an existing
operation unless it creates a genuinely different arbitration.

\subsection{Auxiliary operations}

\texttt{cache.prefetch@1} requests proactive state loading, while
\texttt{request.rebalance@1} requests live movement. The adapter still resolves
source, transfer, allocation, destination capacity, and failure
\cite{sun2024llumnix}. Engines lacking those mechanics omit them during
negotiation. Explicitly separating this tier prevents a portability claim from
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
become \texttt{schedule}; cache pressure becomes \texttt{cache}; progress,
boundary, resume, preemption, and finish become \texttt{feedback}. Active
cache loading maps to optional prefetch action requests, and routing under
sustained imbalance may stage a rebalance action. Idle maintenance remains a
host mechanism. The artifact contains the full fourteen-row crosswalk.

Across [$K$] artifacts, the historical labels cover [96\%] of observed policy
changes, and the remapping preserves [all/$\cdot$\%] of those policies using
five core operations plus negotiated mechanics. Residual artifacts either modify
mechanics such as kernels and parallelism or identify [describe any unmatched
subject]. As a temporal check, a surface frozen on artifacts before [date]
covers [$\cdot$\%] of [$M$] later artifacts. These measurements test both
directions of the factoring: too few operations lose policies; too many merely
rename triggers.

\begin{figure}[t]
\centering
\framebox[\columnwidth]{\parbox[c][4.5cm][c]{0.92\columnwidth}{\centering
\emph{Placeholder --- Figure 2.}\\[2pt]
\footnotesize A request enters through \texttt{admit}; placement,
service, and residency form the three middle control loops; enacted outcomes
flow through \texttt{feedback} into shared, group, and request policy state
read by later decisions. Prefetch, cancellation, swap, and rebalance appear as
negotiated actions outside the core ring.}}
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
operations and share explicit policy state across them. At most one package owns a given
operation in a deployment; automatically composing two independently written
rankers is outside the contract.

Authority is split among four actors. The \emph{application} supplies bounded
but untrusted request fields. The \emph{policy author} declares implemented
operations, required and optional mechanics, schema requirements, and resource
limits. The \emph{operator} attaches the package to principals or workloads and
chooses supported mechanics and schemas. The \emph{adapter and host}
authenticate principals, supply authoritative facts and feasible candidates,
validate results, and enact them through engine mechanics.

Attachment is the approval and linking boundary. The host exact-matches the
contract and package version, checks requested limits, verifies the component
surface, and fails if any required mechanic or schema is unavailable. Optional
mechanics are intersected with host support and supplied in decision metadata;
missing support is never silently emulated.

\subsection{Typed operations}
\label{sec:invocation}

The conceptual SDK presents ordinary typed functions:

\begin{figure*}[t]
\begin{small}
\begin{verbatim}
admit(candidates[], capacity, meta)
    -> decisions[Accept | Defer | Reject]

route(requests[], targets[], feasible_edges[], meta)
    -> decisions[Assign(edge) | Defer]

schedule(runnable[], capacity, meta)
    -> selections[{ requests[], token_budgets[] }]

cache(resident[], prospective[], capacity, episode?, meta)
    -> { admissions[Cache | Bypass], reclaim[] }

feedback(delivery_id, records[]) -> state_update
\end{verbatim}
\end{small}
\caption{The five core signatures. Admission and routing are dense and
input-aligned; scheduling and cache reclaim use explicit validated sets.}
\label{fig:signatures}
\end{figure*}

Dense admission and route results avoid repeating identifiers. The host
validates route edges, target count/resource capacity, schedule overlap and
token budgets, prospective cache admission, reclaim eligibility, and retained
bytes. A wrong length, stale reference, trap, deadline miss, or invalid state
update discards the invocation and exposes no staged actions. Input order,
opportunity ID, retry attempt, and snapshot reference are replay-visible.

Every decision operation is set-oriented. A singleton uses an array of length
one; no separate singleton ABI or batch capability exists. The adapter chooses
bounded opportunity boundaries and must preserve membership and order across a
state-conflict retry.

\subsection{Documents and explicit policy state}
\label{sec:signals}

Structural safety fields---identities, lifecycle status, indices, lists,
variants, capacities, and plans---are typed in WIT. Engine-specific
\emph{facts}, mutable request \emph{fields}, and policy \emph{scratch} remain
bounded JSON objects. Facts are host-owned; fields and scratch preserve their
writer and provenance:

\begin{verbatim}
let served = state.request("R")?
                  .facts()["attained_service"];
state.request_mut("R")?
     .scratch["debt"] = next_debt;
\end{verbatim}

The guest-visible state has three scopes:

\begin{verbatim}
PolicyState {
  shared,
  groups[GroupId] {
    principal, status, limits, member_count,
    facts, scratch
  },
  requests[RequestId] {
    request_ref, status,
    facts, fields, scratch
  }
}
\end{verbatim}

Request-to-group membership, principal, lifecycle status, quotas, facts, and
working-set membership are immutable to the guest. The SDK exposes group and
request facts read-only and computes a sparse update containing only changed
shared state, group scratch, request fields, and request scratch. Each listed
document is a complete namespace replacement; v0.6 does not define JSON Patch
or ambient state calls.

\begin{verbatim}
StateUpdate {
  shared?,
  groups[{ group_id, scratch }],
  requests[{ request_id, fields?, scratch? }]
}
\end{verbatim}

The host derives the exact working set from the validated context. It includes
shared state, every referenced request, and each referenced request's trusted
work group exactly once. It does not implicitly load siblings. Shared, group,
and request scopes have host-private revisions; group scratch and member count
are quota-bounded.

The following abbreviated scheduler uses typed structure and extensible facts:

\begin{figure}[t]
\begin{small}
\begin{verbatim}
fn schedule(cands, cap, ctx) {
  let i = argmin(cands, |r| {
    state.group(r.group_id)
         .scratch["service"]
  });
  SchedulePlan {
    selections: [{
      requests: [i],
      token_budgets: [
        min(cands[i].max_budget,
            cap.remaining_tokens)
      ]
    }]
  }
}
\end{verbatim}
\end{small}
\caption{A policy reads trusted group state and returns an explicit selection.}
\label{lst:policy}
\end{figure}

PLEX provides conditional \emph{policy-state commit}, not an external engine
transaction. The host loads one coherent working set, invokes the component,
validates the plan and update, compares revisions, commits policy state, and
only then exposes the plan and staged actions to the adapter. The adapter
revalidates and enacts separately. Enactment failure does not roll back policy
state, so decision hooks may record intent or attempt state only;
success-dependent accounting belongs in \texttt{feedback}. A feedback delivery
ID, its state update, terminal cleanup, and deduplication record commit
together.

\begin{proposition}[Observed substitution is strategyproof]
\label{prop:sp}
Consider a scheduler that orders requests only by cumulative
host-observed attained service and serves the least-served first. If tenants
can choose metadata but cannot alter observed service, allocation is
independent of every declaration. Truthful declaration or omission is
therefore weakly dominant.
\end{proposition}

Provenance does not impose this policy; it makes such defenses expressible.
A mixed policy may compare predicted with realized length in
\texttt{feedback}, store credibility in principal-scoped state, and discount
repeated misdeclaration.

\subsection{An ABI that can grow}
\label{sec:portability}

The structural WIT surface is closed and exact-versioned: changing a record or
variant advances the component contract. Extensible facts, fields, scratch,
query arguments, and action arguments remain bounded JSON documents governed
by independently versioned schemas and method names. Package format, WIT
contract, engine JSON API, and helper methods have separate version axes.

Required and optional dependencies have only two outcomes. A missing required
mechanic or schema rejects attachment. A missing optional mechanic is absent
from negotiated decision metadata, and an attempted call returns an explicit
unsupported-mechanic failure. PLEX does not silently coerce an action to a
``nearby'' mechanic.

The host materializes the complete typed context and policy-state working set
before crossing the sandbox, so field access requires no host calls. The only
imports are immediate read-only \texttt{query} and staged \texttt{action};
their counts and aggregate bytes are metered and bounded.

\subsection{One request end to end}
\label{sec:worked}

Consider a tool-calling request in a ten-step workflow. Its first
generation arrives with a workflow ID and expected output length as metadata.
\texttt{admit} accepts it against host load and operator limits.
\texttt{route(Arrival)} combines a preferred-region hint with observed prefix
overlap and queueing, then selects a feasible replica. \texttt{schedule}
combines the tenant's operator weight, request or work-group attained service, and
learned length state to assign service and a token budget.

After each engine step, one batched \texttt{feedback(Progress)} adds committed
tokens and service time to request or work-group state. The model emits a tool call.
\texttt{feedback(Boundary)} records a bounded retention intent keyed by the
request; if the policy reserves future service, it writes a standard
action or bounded scratch record with a host-enforced TTL. While the tool runs,
\texttt{cache} sees resident units annotated with enacted host facts and
retention value and reclaims lower-value state.

The follow-up presents the continuation capability, creating a new generation
under the same request. It is not admitted again.
\texttt{route(Arrival)} now sees where its KV remains resident, and
\texttt{schedule} sees accumulated service rather than resetting fairness.
On completion, \texttt{feedback(Finished)} performs terminal accounting and
removes outstanding intent entries. The hooks coordinate through scoped state,
but every physical action remains an engine transition.

\section{The System}
\label{sec:system}

PLEX consists of a policy package, a generic host, and thin router and engine
adapters (Figure~\ref{fig:arch}).

\begin{figure}[t]
\centering
\framebox[\columnwidth]{\parbox[c][4.8cm][c]{0.92\columnwidth}{\centering
\emph{Placeholder --- Figure 3.}\\[2pt]
\footnotesize Application fields and operator-selected schemas/mechanics enter
a PLEX host. Router and engine adapters provide trusted identities, scoped
state, facts, and feasible sets; typed direct plans return to native mechanics.
Conditional state commit connects hooks, while enacted outcomes return through
feedback.}}
\caption{PLEX architecture. The contract carries policy and typed data;
adapters retain engine-specific mechanics.}
\label{fig:arch}
\end{figure}

\subsection{Host and hot path}

The host instantiates a Wasm component with one PLEX import interface:
versioned, synchronous read-only \texttt{query} and staged \texttt{action}.
It exposes no WASI, filesystem, socket, arbitrary engine object, or ambient
state-loading API. At attachment it validates package format 6, the exact
\texttt{pie:plex@0.6.0} surface, limits, schemas, and required mechanics.

For each invocation, the adapter produces one bounded typed context. The host
derives and loads the exact state working set, charges input bytes against
policy limits, and enters Wasm once. The direct plan, sparse state update, and
staged actions are validated before any of them become visible to the adapter.

\subsection{Isolation, availability, and policy quality}
\label{sec:safety}

Three guarantees must not be conflated. \emph{Mechanical isolation} comes
from Wasm memory isolation and a typed host surface: policy code cannot mutate
engine objects or request contents. \emph{Availability} comes from deadlines,
bounded allocation, state quotas, and engine-default fallback. A
trap, timeout, invalid plan, or invalid state update discards the invocation's
state changes and staged actions and applies the native heuristic.

\emph{Policy quality} is not guaranteed by sandboxing. A valid ranker may make
poor decisions or favor one tenant. The host bounds effects it owns:
deferral is capped, host-consumed intents expire and consume quotas, and a
progress watchdog may detach a package whose workloads stop advancing.
Beyond those bounds, operators remain responsible for the policy they attach.

Telemetry follows a separate lossy path. Policies may emit bounded metrics or
debug records to a ring buffer, but no decision or policy-state commit depends on
telemetry delivery.

\subsection{Conditional state commit and replacement}

Each invocation receives immutable facts and one coherent shared/group/request
snapshot. The host validates sparse replacements and compares every scope
revision in the working set. A conflict commits neither state, actions,
feedback deduplication, nor cleanup; the adapter may retry the same opportunity
with an incremented attempt and a fresh snapshot. On success, policy state
commits before the normalized plan and staged actions are exposed. The adapter
then revalidates and enacts separately, and later feedback records actual
outcomes. PLEX therefore provides process-local policy-state atomicity, not
two-phase or crash-atomic coordination with external engine state.

Attachment creates a link between one package manifest owner and each operation it
owns. Replacing a link is atomic: new calls enter the new component while
in-flight calls finish under the old one. A failed activation leaves the prior
link intact, and detach restores the engine default.

\subsection{Adapters and attachability}
\label{sec:impl}

Adapters translate semantic views rather than expose engine structures.
Routing usually attaches at a cluster router. Admission and scheduling attach
where an engine accepts and selects work. Cache policy attaches where the
engine can present prospective objects and legal reclaim units. Feedback taps
committed execution and lifecycle changes.
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
\textbf{cache} & \textbf{feedback} & \textbf{prefetch action} &
\textbf{rebalance action} & \textbf{Adapter LOC} \\
\midrule
vLLM & router & [N] & [N/A] & [N/A] & [N] & [N] & [--] & [418] \\
SGLang & router & [N] & [N/A] & [N/A] & [N] & [N] & [--] & [$\sim$250] \\
TensorRT-LLM & router & [N] & [N/A] & [N/A] & [N] & [--] & [--] & [$\sim$xxx] \\
Dynamo & [N] & [N] & [N/A] & [--] & [N] & [N] & [N] & [256] \\
Pie & router & [N] & [N] & [N] & [N] & [N] & [--] & [$\sim$xxx] \\
\bottomrule
\end{tabular}
\end{table*}

The current artifact contains the typed interface and validators, a Wasmtime
host, Rust and Python SDKs, vLLM and SGLang adapter templates, 31
evidence-classified policy kernels, negative fixtures, deterministic replay,
and a generated replication report. The harness drives recorded opportunities
through the host and compares normalized plans, fallbacks, scoped state
updates, actions, and feedback effects.

\section{Evaluation}
\label{sec:eval}

We evaluate whether the stable waist captures real policies, transfers across
engines, grows without ABI proliferation, and is practical on serving hot
paths. All bracketed live-workload quantities below remain pending their
engine run; conformance, replication, replay, and validator benchmarks reported
by the repository are complete.

\paragraph{Setup.}
[Models, GPUs, engines and versions, trace sources, policy packages, policy
state configuration, and workload-generation parameters.]

\subsection{E1: Expressiveness}
\label{sec:eval-expressiveness}
\label{sec:eval-q3}

\paragraph{Corpus remapping.}
We first apply the fixed fourteen-to-five protocol of
\S\ref{sec:elbow} to all [$K$] artifacts. Report (i) historical-grid
coverage, (ii) the fraction preserved by the five core operations, (iii) the
increment from negotiated standard mechanics, and (iv) residuals that change
mechanics. A merge/split sensitivity analysis tests whether another operation
boundary yields the same coverage with less authority.

\paragraph{Auditing policy adaptations.}
We implement all 31 candidates in the committed matrix. Each has source-linked
paper metadata, a component, a deterministic smoke case, an expected result,
and named deferred mechanics. Four independent fidelity reviews found no
faithful or faithful-with-deferred-mechanics reproduction: 17 have material
semantic gaps and 14 implement a different core algorithm. We therefore label
all 31 as inspired adaptations until pinned paper/artifact differential traces
pass. The executable performance harness reports only paper-anchored proxy
trends and never equates those ratios with the papers' end-to-end results.

\paragraph{Coordinated policy.}
A single package implements admission, routing, scheduling, cache policy, and
feedback around a workflow. Routing records enacted locality in policy state;
feedback records progress and bounded continuation intent; later schedule and
cache calls consume that state. We compare engine defaults, the best
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
The replay harness feeds canonical opportunities and scope revisions to each
adapter and compares direct plans, state updates, actions, feedback effects,
and classified failures.
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
\item add a namespaced fact schema and run an old binary unchanged;
\item add an optional action schema and verify explicit absence when unsupported;
\item add a cache-object kind through extensible facts;
\item omit a required mechanic and verify attach-time rejection;
\item replace a package atomically while preserving backend policy state; and
\item share group/request accounting across \texttt{feedback},
  \texttt{schedule}, and \texttt{cache}.
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
Measure candidate materialization, group auto-join, sandbox crossing, policy
execution, result validation, and commit separately. Report per-hook latency
versus candidate count; batched \texttt{feedback} cost versus records per call;
state load/update/conflict cost; and end-to-end throughput, TTFT, and token
latency. On the development x86-64 host, committed release-profile validator
medians are 0.8\,$\mu$s for singleton admission, 28.6\,$\mu$s for 64-request
admission, 43.5\,$\mu$s for 64-by-8 routing, 54.1\,$\mu$s for 128-request
scheduling, and 158.8\,$\mu$s for 1,024 cache objects. Live-engine overhead
remains a separate evaluation.

\paragraph{Failure behavior.}
Inject traps, infinite loops, dense-length errors, out-of-range indices,
oversized token budgets, unauthorized actions, malformed documents, and
unknown state scopes. Verify native fallback, policy-state/action rollback, and
continued progress; report detection latency and fallback rate.
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
The typed contract, scoped state, package replacement, standard action
validation, deterministic replay, 31 inspired policy adaptations, and a
version-pinned asynchronous vLLM integration are implemented. Physical
swap/migration/prefetch enactment, cluster routing/admission, control-plane
provisioning, predictor training, and original paper-scale workloads remain
outside the current evidence and are named explicitly.

\section{Related Work}
\label{sec:related}

\paragraph{Context-aware serving.}
Program fairness, workflow-aware cache policy, tool continuation, affinity
routing, and critical-path scheduling motivate PLEX
\cite{sheng2024fairness,abhyankar2024infercept,luo2025autellix}.
Parrot carries application structure through semantic
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

Agentic applications changed serving from isolated generations into requests
that pause and continue and work groups containing independently executing
siblings. The information needed to govern those subjects lives above engines,
while the authority to admit, place, schedule, and retain state lives inside
them. Metadata alone cannot bridge the gap when policy remains fixed, and
forks do not provide a stable research or deployment interface.

PLEX programs the request lifecycle through a five-operation closed loop:
admission, three resource arbitrations, and feedback. Typed structure preserves
identity and feasibility; explicit shared/group/request state carries learned
state and bounded intent; negotiated actions preserve the mechanism boundary;
adapters retain feasibility and mechanics. The result is intended to do for serving policy
what successful extensible substrates do elsewhere: keep the authority waist
small while allowing events, data, policies, and implementations to evolve
around it.

\bibliography{references}
\bibliographystyle{mlsys2025}

\end{document}
