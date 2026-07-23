# Drift Analysis (Phase 6)

## Scope

This document summarizes the written interpretation of the Phase 6 drift-monitoring evidence already produced in this repository. It is intended as an engineering and interview-ready explanation of what the drift signals mean and what they do not mean.

## Observed results

- Stable batch result: pass. The stable batch is a deterministic row-order permutation control and is expected to pass the configured drift gate.
- Drifted batch result: fail. The deliberate drifted batch is expected to breach the configured gate and return a non-zero monitoring exit.
- Dataset-level drift outcome (drifted batch): drift detected.
- Drifted feature count: 5 of 30 evaluated features (16.67%).
- Drifted features:
  - Age
  - MonthlyIncome
  - TotalWorkingYears
  - JobRole
  - OverTime

## Feature-level interpretation

| Drifted feature | Likely business interpretation | Possible model impact |
|---|---|---|
| Age | Workforce age mix changed (for example, more early-career or late-career hires). | Decision boundaries learned from historical age distribution may become less calibrated for the new mix. |
| MonthlyIncome | Compensation bands shifted due to hiring mix, market adjustments, or policy updates. | Income-related splits or coefficients may represent older pay distributions and reduce score stability. |
| TotalWorkingYears | Overall tenure/seniority profile changed. | Experience-related signal strength can change, affecting attrition risk ranking consistency. |
| JobRole | Role composition changed (new role mix or different team structure). | Category frequencies and one-hot patterns can shift, making some learned role effects less reliable. |
| OverTime | Overtime incidence changed operationally. | If overtime was a strong historical indicator, its predictive contribution may drift in magnitude. |

## Why drift does not automatically mean model failure

- Drift is a data-distribution signal, not a direct measure of model quality.
- A model can remain acceptable under moderate drift if the shifted features have limited decision influence or if relationships remain stable.
- Confirming model health still requires outcome-based checks such as recent labeled performance, calibration, and threshold monitoring.

## Simulated validation drift vs. real production drift

- Simulated drift in this repository:
  - Purpose: deterministic validation of monitoring logic and CI gate behavior.
  - Behavior: intentionally modified feature distributions while preserving safe controls (for example, no raw CSV mutation and no target mutation).
- Real production drift:
  - Purpose: operational risk detection in live data.
  - Behavior: naturally occurring shifts from business, workforce, policy, or market changes.

A simulated drift failure validates the monitor implementation; it does not claim that the production model has already degraded.

## Recommended follow-up actions

1. Continue scheduled drift checks and retain JSON/HTML artifacts for trend review.
2. Add rolling-window drift baselines (for example weekly/monthly) to distinguish one-off spikes from sustained shifts.
3. Prioritize label collection for drifted segments to verify whether predictive quality changed.
4. Define escalation thresholds that combine drift signals with performance signals before retraining decisions.
5. Review feature ownership with HR/business stakeholders for context on compensation, role, and overtime policy changes.

## Current monitoring limitations

- Current evidence is deterministic and scenario-based; it is not live production telemetry.
- The gate is threshold-driven and does not, by itself, estimate business impact or error-rate impact.
- No automatic retraining trigger is configured.
- Without fresh labeled outcomes from production, model-performance impact remains a hypothesis to validate, not a confirmed degradation.
