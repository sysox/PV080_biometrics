# 🛠️ Implementation Plan — Interactive Fingerprint Pipeline

## 1. Goal

Implement a **generic interactive pipeline engine** for fingerprint processing that supports:

* reusable step definitions
* notebook-specific lightweight / advanced presets
* efficient recomputation
* parameter tuning and export
* later reuse of tuned parameters as defaults

---

## 2. Target File Structure

```text
src/
├── interactive.py
├── pipeline_presets.py
└── ...
notebooks/
├── fingerprint_scan_student.ipynb
├── fingerprint_photo_home.ipynb
└── ...
```

### Responsibilities

#### `src/interactive.py`

Generic framework only:

* `PipelineContext`
* `Step`
* `InteractivePipeline`
* execution / caching / UI logic
* parameter export/import

#### `src/pipeline_presets.py`

Preset definitions:

* operation registry
* widget factory
* scan preset
* photo preset
* helper constructors

#### notebooks

Minimal orchestration only:

* load image
* choose preset
* create pipeline
* display pipeline
* optionally save tuned params

---

## 3. Phase 1 — Strengthen Core API

## 3.1 Update `Step` model

Extend `Step` to support explicit input mapping.

### Current issue

`input_keys` alone is too weak for larger pipelines.

### Required

Use:

```python
input_map = {
    "img": "image",
    "mask": "mask"
}
```

instead of only positional / heuristic mapping.

### Step should contain

* `name`
* `func`
* `input_map`
* `output_keys`
* `params`
* `enabled`
* `description`
* cached execution state

### Cached execution state

Each step should store:

* `cached_outputs`
* `last_input_signature`
* `last_param_signature`
* `last_enabled`
* `is_dirty`

---

## 3.2 Improve `PipelineContext`

Keep context as shared dict-like storage.

### Context must support keys like:

* `original`
* `image`
* `mask`
* `binary`
* `skeleton`
* `minutiae`
* `orientation`
* `frequency`

### Requirement

Steps must be able to consume outputs from **any previous step**, not only the immediately previous one.

---

## 3.3 Extend `InteractivePipeline`

Add support for:

* dirty propagation
* context snapshots
* partial rerun from changed step onward
* parameter export/import
* preset loading

---

## 4. Phase 2 — Efficient Execution

## 4.1 Problem to fix

Current pipeline behavior recomputes all steps from the beginning after every widget change.

That is inefficient.

---

## 4.2 Required behavior

### Incremental recomputation

If step `k` changes:

* steps `< k` remain valid
* restore context from snapshot after step `k-1`
* recompute only `k ... end`

---

## 4.3 Context snapshots

Pipeline should maintain:

```python
context_snapshots[i]
```

where snapshot `i` means context state **after step i**.

### Use

If step 4 changes:

* restore snapshot after step 3
* rerun from step 4

---

## 4.4 Dirty propagation

When:

* a parameter changes
* step enabled/disabled changes

then:

* mark that step dirty
* mark all downstream steps dirty

---

## 4.5 Suggested execution flow

1. user changes widget in step `k`
2. step `k` notifies pipeline
3. pipeline marks `k..end` dirty
4. pipeline restores context snapshot from `k-1`
5. pipeline reruns only dirty suffix
6. pipeline refreshes displays

---

## 5. Phase 3 — UI Architecture

## 5.1 Layout modes

Implement two display modes.

### Fixed Grid ("Laboratory")

* all defined steps visible
* disabled steps shown as placeholders
* stable visual pipeline map

### Dynamic ("Experiment")

* only enabled steps shown
* layout reflows automatically
* supports minimal pipelines

---

## 5.2 Step widget design

Each step widget should contain:

### Visualization row

* optional input thumbnail
* output image
* optional histogram / overlay toggle

### Controls row

* enable checkbox
* sliders / dropdowns
* optional save/export intermediate button

---

## 5.3 Separate execution from layout

Changing layout mode should:

* redraw UI only
* not trigger recomputation

---

## 6. Phase 4 — Preset System

Create `src/pipeline_presets.py`.

---

## 6.1 Operation registry

Define one registry of available operations.

Each operation should specify:

* display name
* function
* input mapping defaults
* output keys
* optional description

---

## 6.2 Widget factory

Define a helper that converts parameter specs into ipywidgets.

Supported parameter kinds:

* `int`
* `float`
* `bool`
* `choice`

---

## 6.3 Preset definitions

Each preset should define:

* step order
* enabled/disabled defaults
* slider ranges
* default values

---

## 6.4 Required presets

### A. Scan student preset

Goal:

* good-quality scans
* fewer operations
* safer defaults

Typical steps:

1. segmentation
2. apply mask
3. CLAHE
4. binarization
5. morphology
6. skeletonization
7. minutiae extraction

### B. Photo home preset

Goal:

* noisy photos
* wider experimentation

Typical steps:

1. grayscale
2. CLAHE
3. enhancement options
4. binarization
5. inversion
6. morphology
7. skeletonization
8. optional minutiae

---

## 7. Phase 5 — Notebook Construction

Keep notebooks thin.

---

## 7.1 Scan notebook

Notebook should:

1. load scan
2. create pipeline from scan preset
3. display pipeline
4. optionally export tuned params

---

## 7.2 Photo notebook

Notebook should:

1. load photo
2. create pipeline from photo preset
3. display pipeline
4. optionally export tuned params

---

## 7.3 Optional third notebook

Parameter tuning notebook:

* same engine
* focus on testing variants
* export best settings

---

## 8. Phase 6 — Parameter Export / Import

This is required because tuning results should be reusable later.

---

## 8.1 Add methods to pipeline

* `get_params()`
* `set_params(config)`
* `export_params_json(path)`
* `import_params_json(path)`

---

## 8.2 Export format

Store:

* preset name
* step enabled state
* current widget values

This allows:

* per-image tuning export
* reusable defaults for future notebooks

---

## 8.3 Use cases

### Student workflow

* tune params
* save config

### Teacher / research workflow

* compare saved configs
* choose robust defaults
* update preset definitions

---

## 9. Phase 7 — Optional Quality-of-Life Features

These are useful but not first priority.

### Possible additions

* save intermediate image from selected steps
* overlay toggle for mask / skeleton / minutiae
* display metrics panel
* compare two parameter presets
* duplicate pipeline for side-by-side comparison

---

## 10. Recommended Order of Implementation

## Step 1

Refactor `interactive.py` core API:

* add `input_map`
* add step dirty state
* add step cache fields

## Step 2

Implement incremental recomputation:

* dirty propagation
* context snapshots
* rerun from changed step onward

## Step 3

Add parameter serialization:

* `get_params`
* `set_params`
* JSON export/import

## Step 4

Create `pipeline_presets.py`:

* widget factory
* operation registry
* scan preset
* photo preset

## Step 5

Create two notebooks:

* lightweight scan notebook
* wider photo notebook

## Step 6

Polish UI:

* fixed / dynamic layout
* placeholders for disabled steps
* optional overlays

---

## 11. Deliverables

### Core code

* updated `src/interactive.py`
* new `src/pipeline_presets.py`

### Notebooks

* `fingerprint_scan_student.ipynb`
* `fingerprint_photo_home.ipynb`

### Optional docs

* `PIPELINE_INTERACTIVE_SPEC.md`
* `PIPELINE_PRESETS_GUIDE.md`

---

## 12. Definition of Done

The implementation is complete when:

* notebook can define subset of operations through presets
* changing a late step does not recompute the whole pipeline
* scan notebook is lightweight and stable
* photo notebook supports broader experimentation
* tuned parameters can be exported and later reused
* generic engine remains independent from fingerprint-specific presets
