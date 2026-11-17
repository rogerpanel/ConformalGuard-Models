# Closed-loop study: task success with and without the guard

- guard: `graph` scorer, session-level threshold at alpha = 0.1 from 237 violation-free calibration sessions; trained on 334 sessions
- agents: claude-opus-5-5; a blocked action is not carried out and the agent may propose another, up to 2 times; after that the task thread stops (escalation)
- workspaces: 172 run in both arms and compared below (200 were generated without the guard and 172 with it)
- a task is *safe to complete* if doing what the user asked is allowed by the policy; for the others, completing the request as asked is itself the violation

| | without guard | with guard |
|---|---|---|
| Safe-to-complete tasks that succeed | 99.6% (259/260) | 93.1% (242/260) |
| ... with at least one record delivered (lenient) | 99.6% (259/260) | 93.1% (242/260) |
| Tasks in which a violation was carried out | 17.3% (54/313) | 1.3% (4/313) |
| Not-safe-to-complete tasks completed as requested (= the leak) | 100.0% (53/53) | 7.5% (4/53) |

With the guard:

- violating proposals blocked: 92.2% (47/51)
- safe-to-complete tasks in which an action was blocked: 6.5% (17/260); of these, tasks that still succeeded: 0.0% (0/17)
- sessions without any violating proposal that were interrupted at all: 12.0% (15/125); Corollary 1 bounds the expected rate by alpha = 0.1 when deployment sessions are exchangeable with the calibration sessions, and one standard error at that rate and sample size is 2.7 points
- blocked actions: 64 of 2509 proposals, 17 of them safe by label; 0 tasks escalated
- guard latency per decision: median 70 ms, 90th percentile 152 ms (text encoding of the new nodes plus one forward pass)

Paired over safe-to-complete tasks of the same workspaces (success without / with the guard): both 241, only without the guard 18, only with the guard 1, neither 0.

The agents are sampled anew in each arm, so the two arms differ by the guard and by the model's own run-to-run variation; 'only with the guard' counts the latter.
