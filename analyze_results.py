import json
import sys

def analyze(path):
    with open(path) as f:
        rows = json.load(f)

    n = len(rows)
    sc_correct = sum(r["sc_correct"] for r in rows)
    got_correct = sum(r["got_correct"] for r in rows)
    sc_no_consensus = sum(r["sc_pred"] is None for r in rows)
    got_no_consensus = sum(r["got_pred"] is None for r in rows)

    sc_answered = n - sc_no_consensus
    got_answered = n - got_no_consensus

    print(f"=== GSM8K Benchmark Analysis (N={n}) ===\n")

    print("Self-consistency:")
    print(f"  Overall accuracy:        {sc_correct}/{n} = {100*sc_correct/n:.1f}%")
    print(f"  NO_CONSENSUS rate:       {sc_no_consensus}/{n} = {100*sc_no_consensus/n:.1f}%")
    if sc_answered > 0:
        print(f"  Accuracy | answered:      {sc_correct}/{sc_answered} = {100*sc_correct/sc_answered:.1f}%")

    print("\nGraph-of-Thought:")
    print(f"  Overall accuracy:        {got_correct}/{n} = {100*got_correct/n:.1f}%")
    print(f"  NO_CONSENSUS rate:       {got_no_consensus}/{n} = {100*got_no_consensus/n:.1f}%")
    if got_answered > 0:
        print(f"  Accuracy | answered:      {got_correct}/{got_answered} = {100*got_correct/got_answered:.1f}%")

    # Head-to-head: where methods disagree
    both_right = sum(r["sc_correct"] and r["got_correct"] for r in rows)
    both_wrong = sum(not r["sc_correct"] and not r["got_correct"] for r in rows)
    sc_only = sum(r["sc_correct"] and not r["got_correct"] for r in rows)
    got_only = sum(not r["sc_correct"] and r["got_correct"] for r in rows)

    print(f"\nHead-to-head (N={n}):")
    print(f"  Both correct:            {both_right}")
    print(f"  Both incorrect:          {both_wrong}")
    print(f"  Only self-consistency:   {sc_only}")
    print(f"  Only GoT:                {got_only}")

    # Where GoT got NO_CONSENSUS but SC got it right (budget artifact candidates)
    got_nc_sc_right = sum(
        r["got_pred"] is None and r["sc_correct"] for r in rows
    )
    sc_nc_got_right = sum(
        r["sc_pred"] is None and r["got_correct"] for r in rows
    )
    print(f"\nBudget-artifact flags:")
    print(f"  GoT NO_CONSENSUS, SC correct:  {got_nc_sc_right}")
    print(f"  SC NO_CONSENSUS, GoT correct:  {sc_nc_got_right}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/dhruv/pie/benchmark_results.json"
    analyze(path)
