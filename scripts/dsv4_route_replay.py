"""Replay a `PIE_ROUTE_DUMP` route trace per slab under LRU, clock, LFU and Belady at several seat counts: `python3 scripts/dsv4_route_replay.py routes.tsv`."""
import sys, collections
lines=[l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
# The dump holds two loads back to back (run-1, run-2); the slab index restarts at 0 each fire.
# Each cut = one line per token row. Group by slab in order.
per={}
for slab, ids in lines:
    ids=[int(x) for x in ids.split() if int(x)>=0]
    per.setdefault(int(slab),[]).append(ids)
E=256
def sim(seq, S, policy):
    """seq: list of cuts (list of ids). Returns misses. Initial seats = identity 0..S-1."""
    misses=0
    if policy=="belady":
        # next-use index per expert
        nxt=collections.defaultdict(list)
        for t,ids in enumerate(seq):
            for e in set(ids): nxt[e].append(t)
        ptr={e:0 for e in nxt}
        seats=set(range(S))
        for t,ids in enumerate(seq):
            want=set(ids)
            for e in want:
                ptr[e]+=1  # consumed this use
            for e in want:
                if e in seats: continue
                misses+=1
                # evict the seated expert not wanted now whose next use is farthest
                cand=[x for x in seats if x not in want]
                def nextuse(x):
                    l=nxt.get(x,[]); i=ptr.get(x,0)
                    return l[i] if i<len(l) else 10**9
                victim=max(cand,key=nextuse)
                seats.remove(victim); seats.add(e)
        return misses
    if policy in ("lru","lfu","lrfu"):
        seats=collections.OrderedDict((e,0) for e in range(S))  # insertion order = recency
        freq=collections.Counter()
        for t,ids in enumerate(seq):
            want=set(ids)
            for e in want: freq[e]+=1
            for e in want:
                if e in seats:
                    seats.move_to_end(e); continue
                misses+=1
                cand=[x for x in seats if x not in want]
                if policy=="lru": victim=cand[0]
                elif policy=="lfu": victim=min(cand,key=lambda x:(freq[x],list(seats).index(x)))
                else:  # lrfu: score = freq * decay by recency rank
                    order={x:i for i,x in enumerate(seats)}
                    victim=min(cand,key=lambda x: freq[x]*(0.9**(len(seats)-order[x])))
                del seats[victim]; seats[e]=0
        return misses
    if policy=="clock":
        in_seat=list(range(S)); used=[False]*S; hand=0; where={e:i for i,e in enumerate(in_seat)}
        for ids in seq:
            want=set(ids); pinned=[False]*S
            for e in want:
                if e in where:
                    used[where[e]]=True; pinned[where[e]]=True; continue
                misses+=1
                for _ in range(2*S):
                    s_=hand; hand=(hand+1)%S
                    if pinned[s_]: continue
                    if used[s_]: used[s_]=False; continue
                    break
                old=in_seat[s_]
                if old is not None and old in where: del where[old]
                in_seat[s_]=e; where[e]=s_; used[s_]=True; pinned[s_]=True
        return misses
ncuts=sum(len(v) for v in per.values())
print(f"slabs {len(per)}, cuts {ncuts}, cuts per slab ~{ncuts//len(per)}")
for S in (40,48,64,96):
    row=[]
    for pol in ("clock","lru","lfu","lrfu","belady"):
        m=sum(sim(seq,S,pol) for seq in per.values())
        row.append(f"{pol} {m/(ncuts/len(per)):6.1f}")
    print(f"seats {S:3d}: misses per token-step:  " + "   ".join(row))
