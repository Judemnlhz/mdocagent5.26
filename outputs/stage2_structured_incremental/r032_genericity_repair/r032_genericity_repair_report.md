# R032 Genericity Repair Report

## Scope

R032 repairs Stage 2 genericity before any further activation or QA scan. It does not run QA, graph expansion, reranking, or effectiveness evaluation.

## Changes

- Removed the deterministic numeric fallback from the default Stage 2 write path.
- Deleted probe-specific numeric extraction helpers that targeted BestBuy-style financial tables and performance-table wording.
- Reworded the document-generic Stage 2 prompt to use domain-neutral table, chart, measurement, count, percentage, date, total, and specification cues.
- Replaced finance/performance lexical quality checks with schema/locator/atomicity checks based on value, row/column/context, and locator completeness.
- Added generic artifact quality tests using non-financial numeric evidence and generic descriptor-table weakness.

## Static Audit

Core files checked:

- scripts/stage2.py
- mdocnexus/stage2/provider.py
- mdocnexus/stage2/artifact_quality.py

Forbidden probe-specific patterns checked:

- BestBuy / bestbuy
- Revenue % change
- Comparable sales
- selected financial data
- Performance Information Table
- deterministic_page_text_numeric_fallback
- _extract_bestbuy_numeric_facts
- _extract_performance_table_numeric_facts

Result: no matches in the checked core Stage 2 files.

## Verification

- python3 -m py_compile scripts/stage2.py mdocnexus/stage2/provider.py mdocnexus/stage2/artifact_quality.py mdocnexus/stage2/tests/test_artifact_schema_validation.py
- python3 -m unittest mdocnexus.stage2.tests.test_artifact_schema_validation
- git diff --check

All completed successfully.

## Decision

Proceed next with a bounded same-page Stage 2 replay after genericity repair. Do not treat R030 deterministic fallback metrics as generic evidence. Activation scan remains blocked until the generic prompt/parser path produces stable atomic artifacts without probe-specific fallback.
