# Revision Plan and Three-Iteration Status

## Immediate Paper Edits Completed

- Included the related-work capability table so `tab:related` resolves.
- Recalibrated unmeasured deployment-extension claims in `Further Considerations`
  and RQ5: unseen brands and new evidence channels are now future-work settings,
  not evaluated capabilities.
- Removed active `\todo{}` markers from the paper body.
- Rewrote cost language as analytical call-count accounting instead of measured
  wall-clock latency.
- Anchored the abstract, introduction contributions, and conclusion to the main
  supported results: 35% AURC reduction, Cov99 from 0.48 to 0.71, and abstention
  on fragile RLWR/ungrounded regions.
- Compactified `tab_related` to fit in one column.
- Shortened and rerendered the risk-coverage plot from the Python generator.

## Requires User Data or New Experiments

- Exact hardware/software stack for the final run.
- Exact `PhishSel` class counts and month-by-month split sizes.
- Source result files behind the populated tables and figure arrays.
- Long-tail-brand and new-channel sweeps if those settings should become evaluated
  claims rather than future work.
- Measured per-stage latency/cost table if the paper should make wall-clock claims.

## Deferred

- The existing notation table still emits an overfull hbox warning. It predates
  this pass and does not affect the experimental-claim cleanup.
- Full bibliography/source-PDF verification remains outside this revision because
  no local source-result bundle was supplied.
