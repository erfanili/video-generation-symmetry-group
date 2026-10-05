# CLAUDE.md — Motion Algebra of Video Generators (CVPR 2027)

## Project in one paragraph
We measure a video generator's motion competence against a known symmetry group, at four nested levels:
what it samples (promptable), what its density covers (supported), what its features represent
equivariantly (equivariant), and what can be steered through those features (steerable).
Testbed: a rigid ball rotating about its own center, so the motion algebra is ℝ³ ⊕ so(3), a 6D twist
ξ = (v, ω). Headline hypothesis: **the training prior restricts sampling, not representation** —
latent steering reaches motions that text prompting cannot.

Deadline: CVPR 2027 paper submission **Nov 16, 2026 AoE**; paper registration ~Nov 10 (verify on OpenReview).

## Framework (the vocabulary every script should use)
- **G / 𝔤**: motion group / Lie algebra. Ball: 𝔤 = ℝ³ ⊕ so(3). Generators `Px, Py, Pz, Jx, Jy, Jz`.
- **μ (twist estimator)**: maps any video → per-frame twist ξ(f). Never reads the prompt.
- **L1 Promptable Π**: twists reached by text / image-to-video sampling, measured by μ.
- **L2 Supported R**: twists the model's density covers (likelihood probe + projection threshold t*).
- **L3 Equivariant**: fitted matrices A_i with h(f+1) ≈ exp(Σ ξ_i A_i Δt) h(f) inside a motion subspace;
  bracket structure tested against candidate algebras.
- **L4 Steerable S**: twists produced by applying exp(Σ ξ_i A_i Δt) to an object's tokens during denoising.
- **Competence gap** Γ = |S \ Π| / |S| over the twist grid.
- **H1 (prior-limited sampling)**: S \ Π ≠ ∅, concentrated off the rolling manifold (ω ⟂ v, |ω|R = |v|).
- **H2 (camera-frame 3D)**: fitted generators best match ℝ³ ⊕ so(3) aligned with camera axes
  (x right, y up, z optical).

### Candidate algebras for model selection (Tab. 3)
| Candidate | [Jx, Jy] | [J_i, P_j] | Pz acts as |
|---|---|---|---|
| aff(2) image-plane | undefined (no out-of-plane tumble) | 2D only | scaling about ball center |
| ℝ⁶ abelian | 0 | 0 | unstructured |
| se(3) rotation about camera | Jz | ε_ijk P_k | scaling about principal point |
| ℝ³ ⊕ so(3) rotation about own center | Jz | 0 | scaling about principal point |

Always compare against null baselines: random-init model, shuffled generator assignments.

## Geometry facts the code must respect
- Image-plane fields: Px, Py constant (image speed ∝ 1/Z ∝ r); Pz is scaling about the **principal point**
  at rate −Ż/Z; Jz is an in-plane rotation on the disk.
- Jx, Jy leave the silhouette unchanged; texture velocity ∝ depth √(1−a²−b²) on the unit disk.
  Linear only on the sphere, not on the image.
- Units: positions in ball radii (X = u/r, Y = v/r, Z ∝ 1/r); v in radii/s; ω in deg/frame.
- Keep |ω| < ~30°/frame. The soccer ball's 5-fold pattern aliases at 72°.

## Models and environment
- Primary: **Wan 2.1 1.3B** (flow matching). Second backbone for generality: CogVideoX-2B or a larger Wan.
- Hardware: lab GPU server via SSH (VS Code Remote-SSH), 120 GB GPUs.
- Tools: SAM2 (silhouette), CoTracker (texture points), Kubric or Blender (renders with exact twists).
- Inversion: use ≥50–100 steps (or forward-noise to a fixed t). Coarse inversion (≈20 steps) is known to
  destroy internal physical signal while still reconstructing the video (Esmati et al. 2026).

## Experiment order (do not skip ahead)
1. **Renders** — soccer ball + die; 12 primitives (6 generators × ±), 60 pairs (15 pairs × 4 sign combos),
   rolling / counter-rolling triples; 2–3 speeds; ground and air; same scene and start pose.
   Cross-content set: basketball, tennis ball, globe, die, same twists.
2. **Validate μ on renders** (Tab. 1): error per twist component; expect ωx, ωy hardest.
   Nothing downstream is trusted until this passes.
3. **L1 prompted probe** (Tab. 2): ~20 seeds × 5 paraphrases per primitive, plus I2V from a fixed first
   frame. Outputs: 6×6 commanded-vs-measured cross-talk matrix, success rate, rolling-coupling rate.
4. **L2 likelihood + projection** (Fig. 2): denoising loss under a neutral prompt across noise levels,
   relative to rolling and to a frame-shuffled control; t* where re-denoising pulls the twist away.
5. **Activation cache**: object tokens only (SAM2 mask), every layer × timestep.
6. **Cross-object twist probe** (soccer ball → globe → die). **Go/no-go Oct 25**: if transfer is weak,
   cut L4 to translation + in-plane spin.
7. **L3**: motion/content principal angles; fit A_i; bracket residuals per candidate algebra; Casimir
   eigenvalues −l(l+1) per irreducible block.
8. **Transplants**: patch motion coords A→B (B moves like A, keeps identity); patch content coords reverse.
9. **L4 steering** (Tab. 4): calibrate gain + linear range per generator; if translation lives in RoPE,
   apply P as positional-phase warp (RoPECraft-style) and report that as a finding. Baselines: text,
   MOFT, DiTFlow, RoPECraft, Tora (trained upper bound). Metrics: twist error, range, identity drift,
   VBench; holonomy test Jx → Jy → Jx⁻¹ → Jy⁻¹ vs predicted Jz spin; two-ball leakage.

## Code conventions
- Every measured quantity is stored with explicit units in the key name (`omega_deg_per_frame`,
  `v_radii_per_s`).
- Every run logs: model id, seed, prompt (or "neutral"), noise level t, inversion steps, layer, git hash.
- Separate **commanded** twist (from render or prompt intent) from **measured** twist (from μ) — never
  mix them in one field.
- Results go to tidy tables (one row per video × frame or per video) so paper tables are a groupby away.
- Fixed random seeds; report mean ± std over seeds.
- Prefer small, verifiable scripts over notebooks; each stage reads the previous stage's saved outputs.

## Paper exhibits to target
Fig. 1 twist-space map (Π / R / S colored, rolling line marked) · Tab. 1 estimator error ·
Tab. 2 prompt cross-talk · Fig. 2 prior map · Tab. 3 algebra selection · Tab. 4 steering vs baselines.
CVPR framing: tables with baselines up front; the algebra presented as "which world model fits the
features", math kept to one boxed definition, Casimir analysis in the appendix.

## Related work to position against
- **Closest competitor**: Esmati et al., *The Invisible Hand of Physics* (arXiv 2606.05328) — physical
  plausibility is linearly decodable from Wan/LTX/CogVideoX internals even when outputs fail. We differ:
  continuous 6D twists (not binary), group structure (brackets), causal control in physical units.
- Joseph et al., *Interpreting Physics in Video World Models* (ICML 2026): direction as a circular
  high-dimensional population code — our Casimir / irreducible-block analysis should explain this.
- Kang et al. (ICML 2025): case-based generalization → predicts rolling coupling.
- Physics-IQ (WACV 2026), LikePhys (ICLR 2026), WMReward (CVPR 2026), DiffTrack (NeurIPS 2025),
  AC3D, RoPECraft (NeurIPS 2025).

## Out of scope (for now)
PGL(3) / projective and quadratic fields; non-rigid objects; contact and occlusion beyond the two-ball
leakage test. Camera ego-motion / parallax is a sibling instance of the same framework (Paper B).
