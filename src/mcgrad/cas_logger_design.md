# TL;DR For Reviewers

## What MCBoost will log per surface

| Field | Bento | FBLearner (Fluent2) | Dataswarm (Calipers) | Fallback / unknown |
|-------|-------|---------------------|----------------------|--------------------|
| `usecase_id` | `"mcboost:bento:{notebook_id}"` via `get_notebook_number()` | `"mcboost:fblearner:{model_type_name}"` from `FlowEnviron` | `"mcboost:calipers:{calibration_name}"` from `cas_telemetry_metadata` | `"mcboost:{uuid}"` |
| `ecosystem` | `STATISTICS` | `STATISTICS` | `STATISTICS` | `STATISTICS` |
| `tool` | `MCBOOST` | `MCBOOST` | `MCBOOST` | `MCBOOST` |
| `surface` | `BENTO` | `FBLEARNER` | `DATASWARM` | `TEST` |
| `primary_owner` | User unixname (auto-detected) | Workflow owner unixname (`CHRONOS_JOB_OWNER`) | ICC config oncall (from `cas_telemetry_metadata`) | Auto-detected or `None` |
| `secondary_owners` | — | — | Auto-detected user (`CHRONOS_JOB_OWNER` = `"xim_quant"`) | — |
| `resource_allocation_group` | `CHRONOS_JOB_SECURE_GROUP` if set | `CHRONOS_JOB_SECURE_GROUP` | `CHRONOS_JOB_SECURE_GROUP` | — |
| `engagement_mode` | `DEDICATED` | `PASSIVE` | `PASSIVE` | `PASSIVE` |
| `tool_data` | `MCBoostData(calibrator_class=<class name>)` | `MCBoostData(calibrator_class=<class name>)` | `MCBoostData(calibrator_class=<class name>)` | `MCBoostData(calibrator_class=<class name>)` |
| `whence` | `SANDBOX` (→ `PROD` after validation) | `SANDBOX` (→ `PROD`) | `SANDBOX` (→ `PROD`) | `SANDBOX` |

Each row is a standard logging field defined by CAS telemetry. See below for details.


## Logging hook

All three surfaces (Bento, Fluent2/FBLearner, Dataswarm/Calipers) use the same code path: `methods.MCBoost(...).fit(...)`. The `.fit()` method on `BaseCalibrator` is the universal entry point for all MCBoost usage. This is where the CAS logger call should go.

`.fit()` is the right hook because:
- It represents "a user ran MCBoost on a calibration task" — one event per training run
- Future `tool_data` payloads (performance metrics, training info) are naturally available after fit completes

Since `BaseCalibrator` is OSS-synced, the CAS logger call must be in internal-only code (via `# @oss-disable` or the `internal/` module).

## Open questions

- **Engagement mode**: No auto-detection in the base CASLogger — each tool implements its own. Do we need engagement mode detection, or is surface detection sufficient for our tracking needs?


----------------------------------
----- Details below this line ----
----------------------------------


# Context

## CAS Telemetry

Our org Central Applied Science (CAS) has developed a telemetry system to log usage of our tools. We want to integrate our family of models (MCBoost) into this telemetry.
Here's a code pointer to the main logger class: fbsource/fbcode/cas_telemetry/cas_telemetry_logger.py
Documentation on environment detection: https://www.internalfb.com/code/fbsource/[ca29cd59da959d74d4d4863e7aae4523682c243a]/fbcode/cas_telemetry/IMPLEMENTATION_SUMMARY.md?lines=11%2C77%2C82%2C96%2C104

COde pointers to integration of other teams in the org:
- fbsource/fbcode/ax/fb/telemetry/ax_cas_logger.py
- fbsource/fbcode/clara/telemetry/ce_cas_logger.py

## MCBoost

We want to serve the logging needs of the org but also use this tool for our own telemetry purposes. We will want to log performance metrics and execution information. The extended logging will be a follow up project but keep it in mind.

Currently all of our customers use MCBoost either via fluent2 (FBLearner) or Dataswarm. Dataswarm is all under one system called CBM (classifier based metrics) or "Calipers". The complication with Calipers is that the owner of a calibration configuration is not completely clear. We might need to ask them to add some logic to provide the owning user/oncall to us.

The CBM system is implemented here: https://www.internalfb.com/code/fbsource/[ca29cd59da959d74d4d4863e7aae4523682c243a]/fbcode/dataswarm-pipelines/tasks/si/imp/calibration_service/mcboost/

Our fluent2 integration is implemented here: https://www.internalfb.com/code/fbsource/[a5cbfa566c24f9a4051cdf124cf40569b17b6cfa]/fbcode/fblearner/flow/projects/fluent2/definition/transformers/contrib/multicalibration/mcboost_transformer.py?lines=310%2C330%2C384-385%2C593-594%2C725-726%2C856%2C986%2C1011%2C1017%2C1164-1165%2C1184%2C1203%2C1207%2C1236%2C1253%2C1255-1256%2C1276%2C1279%2C1324%2C1595

Our Bento users just use fbcode/multicalibration directly so that should be straight forward. Most customers likely use a clone of our opportunity sizing notebook template: N6132191. So this is a good example of how MCBoost is used in Bento.


# How the CAS logger works

## What gets logged

Each event writes to Hive/Scuba table `datascience.cas_tool_usage` with these fields:

| Field | Auto-detected? | Description |
|-------|---------------|-------------|
| `usecase_id` | No | Caller-provided identifier for the use case |
| `ecosystem` | No | CAS ecosystem enum (AX, LABELING, EXPERIMENTATION, etc.) |
| `tool` | No | Tool enum — `MCBOOST` already exists as a value |
| `surface` | **Yes** | Execution environment (BENTO, FBLEARNER, DATASWARM, CHRONOS, BUCK) |
| `primary_owner` | **Yes** | User unixname |
| `secondary_owners` | No | Additional stakeholders |
| `resource_allocation_group` | No (helper available) | Secure group / ACL |
| `engagement_mode` | No | DEDICATED (interactive) vs PASSIVE (scheduled) |
| `tool_data` | No | Tool-specific structured payload (union type) |

Auto-detection silently swallows exceptions so logging never blocks tool execution.

## Environment detection (`detect_tool_surface()`)

Priority-ordered chain in `cas_telemetry/metadata_helpers.py`:

1. **Bento**: `bento.lib.metadata.get_notebook_number() is not None`
2. **Dataswarm**: `os.getenv("DATASWARM_TASK_DATA_ANNOTATIONS")` is set
3. **FBLearner**: `os.getenv("FLOW_DRIVER_PORT")` is set OR `os.getenv("FB_PAR_MAIN_MODULE")` starts with `"fblearner"`
4. **Chronos**: `os.getenv("CHRONOS_JOB_INSTANCE_ID")` — checked after FBLearner because FBLearner jobs run on Chronos
5. **Buck**: `os.getenv("FB_PAR_MAIN_MODULE")` is set (any value)
6. **Default**: `Surface.TEST`

## User detection (`get_current_user()`)

Fallback chain:

1. `os.environ.get("CHRONOS_JOB_OWNER")` — covers Chronos/FBLearner
2. `os.environ.get("APP_META_DATA_JSON")` → parse JSON → `"initiator"` field — Dataswarm (because `$USER` is a secure_group there, not the actual user)
3. `employee.get_current_unix_user_fbid()` → `employee.uid_to_unixname()` — Bento / interactive
4. `os.getenv("USER")` fallback
5. `os.environ.get("SUDO_USER")` final fallback

## Resource allocation group

`os.getenv("CHRONOS_JOB_SECURE_GROUP")` — works across Bento, FBLearner, Dataswarm, Chronos.

## How other CAS tools integrate

**Ax** (`ax/fb/telemetry/ax_cas_logger.py`): Subclasses `CASLogger`, hardcodes `Ecosystem.AX`, wraps `AxData` payload. Factory extracts `usecase_id`, owners from Experiment objects.

**CLARA** (`clara/telemetry/ce_cas_logger.py`): Subclasses `CASLogger`, hardcodes `Ecosystem.LABELING`, validates allowed tools. Also implements its own engagement mode detection with per-surface logic.

**SCMTools** (`central_applied_science/SCMTools/utils/cas_logger.py`): Lightweight wrapper function, no subclass. Creates `CASLogger` directly and calls `log_sync()` in model `__init__`.

**BITES** (`cas_experimentation/telemetry_utils.py`): Simple function with a `BitesMetadata` dataclass, creates `CASLogger` and logs.

Common pattern: all integrations call `log_sync()` (not async) and let surface + primary_owner auto-detect.

# Logging hook

All three surfaces (Bento, Fluent2/FBLearner, Dataswarm/Calipers) use the same code path: `methods.MCBoost(...).fit(...)`. The `.fit()` method on `BaseCalibrator` is the universal entry point for all MCBoost usage. This is where the CAS logger call should go.

`.fit()` is the right hook because:
- It represents "a user ran MCBoost on a calibration task" — one event per training run
- All arguments that identify the use case are available: `prediction_column_name`, `label_column_name`, feature column names
- Future `tool_data` payloads (performance metrics, training info) are naturally available after fit completes

Since `BaseCalibrator` is OSS-synced, the CAS logger call must be in internal-only code (via `# @oss-disable` or the `internal/` module).

## Passing metadata to the logger

No signature change to `.fit()`. Callers pass an optional `cas_telemetry_metadata` dict via `**kwargs`:

```python
mcboost.fit(
    df, prediction_col, label_col,
    cas_telemetry_metadata={"primary_owner": "some_oncall", "usecase_id": "my_calibration"},
)
```

The internal logging code pops `cas_telemetry_metadata` from kwargs before the actual training runs. Supported keys:
- `primary_owner` — override for owner (otherwise auto-detected from env)
- `usecase_id` — override for use case identifier (otherwise auto-generated per surface)

For most surfaces this dict is not needed — everything auto-detects. The main consumer is Calipers/Dataswarm, which will pass the ICC config's `oncall` as `primary_owner`. If the dict is absent or a key is missing, the logger falls back to auto-detection (e.g. `CHRONOS_JOB_OWNER` for Dataswarm). This covers future Dataswarm users outside of Calipers.

## Owner field convention

The `primary_owner` and `secondary_owners` fields serve different purposes depending on the surface:

| Surface | `primary_owner` | `secondary_owners` |
|---------|-----------------|-------------------|
| **Bento** | Auto-detected user unixname | — |
| **FBLearner** | Auto-detected workflow owner (unixname) | — |
| **Dataswarm (Calipers)** | Calibration config oncall (from `cas_telemetry_metadata`) | Auto-detected initiating user (`CHRONOS_JOB_OWNER`) |
| **Dataswarm (other)** | Auto-detected pipeline oncall (`CHRONOS_JOB_OWNER`) | — |

For Calipers, there is no meaningful individual user — the oncall that owns the calibration config is the right `primary_owner`. The `secondary_owners` captures the initiating user, which is `"xim_quant"` (pipeline oncall) for scheduled runs but could be an individual for manual backfills.

# Integration analysis per surface

## Bento

Auto-detection works fully out of the box:
- **Surface**: `bento.lib.metadata.get_notebook_number()` returns non-None → `Surface.BENTO`
- **User**: Resolves via `employee.get_current_unix_user_fbid()` → notebook owner's unixname
- **Resource group**: `CHRONOS_JOB_SECURE_GROUP` available if kernel runs on a secured allocation

No complications expected. Bento can rely entirely on CAS logger auto-detection.

## FBLearner (Fluent2)

Auto-detection for surface and user works:
- **Surface**: `FLOW_DRIVER_PORT` or `FB_PAR_MAIN_MODULE` starting with `"fblearner"` → `Surface.FBLEARNER`
- **User**: `CHRONOS_JOB_OWNER` → workflow owner's unixname
- **Resource group**: `CHRONOS_JOB_SECURE_GROUP` available

The Fluent2 MCBoost integration lives in `fblearner/flow/projects/fluent2/definition/transformers/contrib/multicalibration/mcboost_transformer.py`. The `.fit()` call happens inside `BaseMCBoostTransformer.get_model()`.

**Owner/team identification challenge**: The transformer object at `.fit()` time does NOT have access to Domain-level metadata (owner, team, domain name). The `Domain` is constructed with the transformer as an argument, not the other way around — there's no back-reference. Available on `self` is only `self.name` (transformer name, e.g. `"mcboost_MCBoostTransformer"`), which is not globally unique.

However, `runtimecontext.get_flow_environ()` is available at runtime inside FBLearner and provides:
- `workflow_name` — e.g. `"fluent2.publish.integrity.sec_genai.cse_genai_mcboost_groups"`
- `root_workflow_model_type_name` — e.g. `"facebook_integrity_eau_group_mm_genai"`
- `workflow_run_id` — unique per run

This is accessible from our internal logging code without changes to Fluent2 plumbing. `root_workflow_model_type_name` is a good `usecase_id` candidate for FBLearner since it maps to `Domain.model_type` and is globally unique.

No blockers. The CAS logger auto-detection covers surface and user. For `usecase_id`, we can opportunistically read `FlowEnviron` in the internal logging code when running on FBLearner.

## Dataswarm (Calipers/CBM)

Auto-detection for surface and user works:
- **Surface**: `DATASWARM_TASK_DATA_ANNOTATIONS` → `Surface.DATASWARM`
- **User**: `APP_META_DATA_JSON["initiator"]` → pipeline initiator's unixname
- **Resource group**: `CHRONOS_JOB_SECURE_GROUP` available

The Calipers MCBoost integration lives in `dataswarm-pipelines/tasks/si/imp/calibration_service/mcboost/`. The call chain is: Dataswarm pipeline → `PythonOperatorV2` → `train_calibrator()` in `src/pandas_training.py` → `methods.MCBoost(...).fit(...)`.

**`calibration_name` is available at `.fit()` time** — it's passed through `func_kwargs` to the training function and is a globally unique identifier per calibration config (e.g. `"integrity_rotr_v2"`). This is a natural `usecase_id`.

**Oncall/owner gap**: ICC configs have an `oncall` field (from `oncall_short_name.name` in GraphQL), and hardcoded configs have `ONCALL` constants. However, `oncall` is NOT currently forwarded through `func_kwargs` to the training function — it's only used at pipeline build time for DQ check task ownership.

The auto-detected `primary_owner` will always be `"xim_quant"` (the pipeline-level oncall from `GlobalDefaults`) for all Calipers runs, because `CHRONOS_JOB_OWNER` resolves to the pipeline oncall, not the per-calibration oncall. This makes auto-detection insufficient for distinguishing customers — the `cas_telemetry_metadata` override is essential here.

**Options to get the calibration oncall**:
- **Chosen: Minimal Calipers change**: Ask the Calipers team to add `"oncall": self.calibration.get_oncall()` to `func_kwargs` in `builder.py` (two lines — one for single-task at ~line 671, one for multitask at ~line 594). Then `train_calibrator()` passes it via `cas_telemetry_metadata={"primary_owner": oncall}` to `.fit()`. The auto-detected user (`CHRONOS_JOB_OWNER`) goes into `secondary_owners`.




## Resolved Questions

- **Oncall in `primary_owner`**: The CAS telemetry announcement explicitly documents `primary_owner` as "Owner of the usecase, can be an oncall". Oncall names are valid values. No issue.
- **Ecosystem**: `Ecosystem.STATISTICS`
- **Testing**: `whence=WhenceScribeLogged.SANDBOX` (default) keeps logs out of production tables. Use the [Logger Inspector dashboard](https://www.internalfb.com/unidash/dashboard/cas_tool_usage/cas_telemetry_logger_inspector) to verify sandbox logs during development.
- **OSS boundary**: `BaseCalibrator` is OSS-synced. CAS logger calls must be in internal-only code (via `# @oss-disable` or the `internal/` module).
- **usecase_id strategy**: Per-surface approach: Bento = caller-provided or auto-generated from fit() args, FBLearner = `root_workflow_model_type_name` from FlowEnviron, Dataswarm = `calibration_name`. Or unify with a single approach?
