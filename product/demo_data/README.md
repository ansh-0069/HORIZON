# Reviewed Demo Metadata

This directory is the product-layer demo fixture. It is not part of the
protected evaluator default (`./data`). Alongside the supplied advertising
exports, it includes two reviewed metadata artifacts used to demonstrate
cross-channel decision controls without semantic warning noise:

- `source_semantics.csv` declares the demo's common `USD` currency, `UTC`
  daily boundary, platform-attributed measurement convention, and the exact
  source column used as revenue for each platform.
- `campaign_taxonomy.csv` maps every Meta campaign ID in this fixture to a
  reviewed operational campaign type. The underlying Meta export does not
  carry this hierarchy directly.

`review_status=reviewed` is an explicit fixture assertion, not an inference by
the software. For a client deployment, replace both files with client-approved
metadata and keep `reviewed` only after the data owner has verified the
currency, timezone, attribution definition, revenue field, and campaign
taxonomy. A missing or unreviewed file remains allowed by the evaluator path,
but the pipeline will expose the corresponding quality warning.
