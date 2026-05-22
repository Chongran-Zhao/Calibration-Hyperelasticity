# DESIGN.md

## Product

**Calibration for Hyperelasticity** is a web-first research tool for fitting
hyperelastic material models to experimental data and running predictions.

The interface should feel like a calm scientific workstation: precise,
legible, trustworthy, and efficient. It should not feel like a marketing page,
demo dashboard, or decorative visualization.

## Design Direction

The interface follows the Stitch-generated web shell direction:

- Light neutral workspace background
- White grouped panels
- Blue primary actions
- Dense but readable scientific controls
- Large plotting areas as the main visual focus
- Step-based workflow from data selection to prediction
- Sticky top navigation and bottom action bar
- Tailwind-compatible design tokens

The design goal is to make complex calibration workflows feel sequential,
auditable, and low-friction.

## Web UI Target

Target strengths:

- The four-step workflow is clear: Experimental Data, Model Architecture,
  Optimization, Prediction.
- Sidebar progress states help users understand what is available or locked.
- The app should keep a coherent visual language across pages.
- Plot previews give the interface a strong scientific anchor.

Risks to avoid during implementation:

- The default viewport should not show horizontal scrolling at 1300px width.
- Primary action buttons must not truncate.
- Pages should not feel over-framed because cards are nested inside cards.
- Several labels and controls compete for horizontal space.
- The visual hierarchy between section title, workspace title, and group box
  title can be tightened.

## Layout Rules

The app uses a web application shell:

- Left sidebar: fixed workflow navigation
- Top bar: product title, search, settings, help
- Main content: active workspace
- Bottom bar: back/next actions and current status

Rules:

- The main content area must not require horizontal scrolling at 1300px width.
- Primary action buttons should fit their text or use shorter labels.
- Plot areas should receive priority over secondary explanatory panels.
- Long scientific labels should wrap or use compact notation.
- Each workflow step should show only the controls needed for that step.
- Avoid deep nesting of bordered panels.

Recommended minimum window:

```text
1000 x 700
```

Recommended comfortable window:

```text
1300 x 850
```

## Color Tokens

```text
--color-bg-app:         #F5F5F7
--color-bg-surface:     #FFFFFF
--color-bg-subtle:      #F2F2F7

--color-text-primary:   #1C1C1E
--color-text-secondary: #3A3A3C
--color-text-muted:     #6E6E73
--color-text-disabled:  #8E8E93

--color-border:         #E5E5EA
--color-border-strong:  #D1D1D6

--color-primary:        #007AFF
--color-primary-hover:  #0A84FF
--color-primary-down:   #0062CC

--color-error:          #FF3B30
--color-success:        #007AFF
```

## Typography

Use platform-native sans-serif fonts.

Preferred stack:

```text
SF Pro Text, SF Pro Display, Helvetica Neue, Segoe UI, Arial, sans-serif
```

Type scale:

```text
App title:        26px / 32px / 600
Workspace title: 18px / 24px / 600
Panel title:     15px / 20px / 600
Body text:       14px / 20px / 400
Body bold:       14px / 20px / 600
Metric value:    16px / 20px / 600
Helper text:     12px / 16px / 400
```

Rules:

- Do not use oversized hero typography inside tool panels.
- Keep scientific labels readable and compact.
- Use muted text for guidance, not for primary data.
- Avoid negative letter spacing.

## Components

### Sidebar

Purpose: workflow orientation.

Rules:

- Fixed width around 260px.
- Step cards show title, state, and current indicator.
- Locked steps use dashed borders and muted labels.
- Completed steps use primary blue status text.
- Author links use icon buttons only.

### Step Cards

States:

```text
locked
active
complete
available
```

Rules:

- Active step has blue border.
- Locked step has dashed neutral border.
- Current step shows a compact arrow indicator.
- Cards should not resize unpredictably when status changes.

### Primary Buttons

Use for forward workflow actions and major computation actions.

Examples:

```text
Next
Start Calibration
Update Prediction
```

Rules:

- Avoid long labels when horizontal space is constrained.
- Prefer `Next` plus context nearby instead of `Next: Model Architecture`.
- Disabled state must remain legible.

### Cards

Use for meaningful task areas only.

Good examples:

```text
Source
Preview
Solver Settings
Parameters
Prediction Data
```

Rules:

- Avoid cards inside multiple other bordered containers.
- Use spacing and headings before adding another border.
- Keep panel titles short.

### Plot Areas

Plots are primary content.

Rules:

- Plot panels should be visually dominant.
- Axes use themed text and subtle dashed gridlines.
- Save actions sit below plots.
- Empty plots should show a useful placeholder state when possible.

## Page-Specific Rules

### Experimental Data

Primary goal: choose data and preview it.

Layout:

- Left: source, modes, dataset details
- Right: plot preview

Improvements:

- Prevent mode lists from forcing horizontal scroll.
- Keep dataset reference below details.
- Make selected modes easy to scan.

### Model Architecture

Primary goal: build spring model.

Layout:

- Top: model status and add spring action
- Body: spring cards and model controls

Improvements:

- Reduce empty horizontal whitespace.
- Make each spring card self-contained.
- Consider icon buttons for add/remove spring actions.

### Optimization

Primary goal: run calibration and inspect convergence.

Layout:

- Top: metric cards
- Left: solver settings, parameters, log
- Right: calibration plot

Rules:

- Metrics must stay compact.
- Log should not visually dominate the plot.
- Running state should be unmistakable.

### Prediction

Primary goal: reuse fitted parameters and compare predictions.

Layout:

- Left: parameter source and prediction data
- Right: prediction plot

Rules:

- Make parameter provenance clear.
- Manual edits should feel deliberate.
- Prediction status should be visible near the plot.

## Interaction Rules

- Locked steps cannot be clicked.
- Next buttons are disabled until prerequisites are met.
- Validation errors use a red border and clear message.
- Long-running optimization must provide progress feedback.
- Save plot buttons remain disabled until a plot exists.

## Accessibility

- Maintain strong text contrast.
- Do not rely only on color for locked, complete, or current states.
- Ensure controls remain usable at minimum window size.
- Avoid clipped text in buttons and labels.
- Keep focus states visible.

## Implementation Notes

The previous Streamlit and PySide interfaces have been removed. The next UI
should be implemented as a web application that binds to the existing Python
calculation layer.

Recommended structure:

```text
frontend/
backend/
src/
```

Recommended mapping:

```text
Stitch HTML/Tailwind -> frontend visual shell
Existing src modules -> backend computation layer
FastAPI or similar -> bridge between UI and Python routines
```

The web UI should reuse the Stitch HTML structure and Tailwind tokens wherever
possible before inventing new styles.

## Near-Term Polish Checklist

- Remove horizontal scrolling at 1300px width.
- Shorten primary next-button labels.
- Reduce nested bordered panels.
- Improve responsive wrapping for long mode labels.
- Make empty plot states more intentional.
- Tighten spacing between workspace header and content.
