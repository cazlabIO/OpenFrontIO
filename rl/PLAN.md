# RL training and local-play implementation plan

Status: design only. This document does not install dependencies, start training,
rent hardware, create cloud resources, or change gameplay.

## 1. Scope and explicit assumptions

Build one end-to-end system: the real TypeScript game engine produces experience,
a Python/PyTorch PPO learner trains on it, and the saved policy can play against a
human in the existing local browser UI.

The initial experiment uses:

- The Box, **normal map size**, normal human economy/combat/territory victory rules.
- FFA with **10, 50, or 100 policy/human slots**, mixed during training, plus
  built-in nations and tribes. Policy-controlled slots use human-player mechanics;
  nations and tribes retain their actual rules. Record all three counts: the
  default 100-policy game has 13 nations and 400 tribes, up to 513 initial entities.
- Default proposal: `nations="default"` (currently 13 on The Box), with an
  Easy → Medium → Hard opponent curriculum and some nation-free self-play games.
  Nation count and curriculum mixture are experiment constants. **Tribes are
  enabled**, initially `bots=400`, matching the current normal game default and
  independently configurable. Nation-free self-play still includes tribes;
  zero-tribe games are explicit debugging/ablation presets, not the training default.
- One policy action per **simulated second**, configurable in whole engine ticks.
  The engine still advances at its normal 10 ticks per simulated second. Training
  does not wait for wall-clock seconds.
- Deterministic, seeded spawn placement; spawn selection is not learned initially.
- All building choices, timing, placement, and upgrades automated outside the RL
  action budget. No messages or emojis.
- Land attacks, expansion into unowned land, attack cancellation, nuclear strikes,
  and alliances including requests, acceptance, rejection, renewal, and explicit
  **breaking**. Preserve normal consequences of betrayal, including nuke-induced
  alliance breaks.

Nuclear weapons are required: enable atom bombs, hydrogen bombs, MIRVs, missile
silos, and SAM defense with their normal engine rules. PPO decides whether and
when to launch, the weapon type, and the target player. **A separate targeting
model chooses the exact strike location.** No spatial strike-location head is
trained by this PPO policy. Building automation does not launch weapons for it.

Keep the earlier no-water/no-boating requirement by explicitly setting
`waterNukes=false`, not by disabling nukes. Do not apply The Box public-playlist
water-nuke modifier to this preset. Normal blast effects, fallout, interception,
weapon costs, and cooldowns remain enabled. If water-creating nukes are desired
later, that is a separate change requiring naval/connectivity support.

Remaining builder assumption: reuse the existing seeded `NationStructureBehavior`
instead of writing another building strategy. Keep the learner's builder on a
fixed Hard behavior profile, independent of opponent curriculum difficulty; inject
that behavior setting explicitly rather than changing human economy/combat rules.
Do not attach the nation attack/diplomacy/nuke controllers to policy players.
Real nation opponents use their full built-in controller at the chosen difficulty.
Tribes also use their existing controller; neither consumes policy inference or
provides PPO training samples.

Learned spawn/building decisions and naval warfare remain out of scope. This
preset deliberately does not reproduce every default option of a public lobby.

## 2. Code organization and boundaries

Keep RL-specific code and generated run data under `rl/`. Small integration hooks
in the existing engine/client are unavoidable; do not fork the game engine or UI.

```text
rl/
  PLAN.md
  README.md
  pyproject.toml                 # Python dependencies, tooling, CPU/CUDA setup
  uv.lock
  __init__.py
  __main__.py                   # train / evaluate / play / benchmark / checkpoint
  models/
    ppo_small/
      constants.py             # ALL experiment defaults for this model
      model.py                 # model factory and architecture-specific code
      README.md                # purpose, measured parameters, experiment notes
    ppo_large/                 # add only after the small baseline works
      constants.py
      model.py
      README.md
  learning/
    config.py                  # typed configuration definitions, not hidden knobs
    policy.py                  # policy contract and typed outputs
    networks.py                # shared attention/spatial/recurrent building blocks
    ppo.py                     # readable PPO update, independent of game IPC
    rollout.py                 # trajectories, sequence batches, GAE
    rewards.py
    self_play.py
    strike_targeter.py          # narrow contract/adapter to the separate aim model
  runtime/
    protocol.py
    environment.py             # Node subprocess ownership and vector stepping
    train.py
    evaluate.py
    play.py                    # localhost inference service and browser launcher
    checkpoint.py
    tracking.py
    backup.py
  ts/
    bridge/                    # Node entry point, framing, request validation
    game/                      # observations, masks, action adapter, builder, spawns
    browser/                   # development-only local-play integration
  protocol/
    README.md                  # versioned wire contract and tensor conventions
    fixtures/                  # shared TS/Python golden messages
  tests/
    python/
    ts/
  tsconfig.json
  runs/                        # ignored: checkpoints, metrics, resolved configs
```

Design rules:

- Prefer functions, typed dataclasses, TypeScript interfaces, and composition.
  Use one small Python policy `Protocol` for interchangeable policies; PyTorch's
  `nn.Module` already supplies the necessary model base class. No custom abstract
  trainer/environment/storage hierarchy without a concrete need.
- Each model's `constants.py` explicitly assembles a frozen configuration covering
  architecture, PPO, reward weights/scales, game preset, player-count mixture,
  evaluation, and checkpoint cadence. The shared config module defines types and
  validation, not a second location for tunable defaults.
- Avoid duplicated PPO implementations between model folders. Register model
  factories in a small explicit registry, not a plugin-discovery framework.
- Treat the separate strike-targeting model as a composable dependency, not a PPO
  output head. If its implementation lives here, give it its own model folder and
  constants file too; do not assume or implement its learning algorithm in this
  PPO plan. Pin the targeter version/settings for reproducible comparisons.
- Runtime CLI overrides such as device, seed, worker count, and tracking mode are
  saved in the resolved run configuration. Never silently change settings on resume.
- Separate pure observation/reward/mask logic from process, filesystem, and cloud
  side effects. Document tensor shapes, time units, and normalization conventions.
- Pin a supported Python version, initially 3.12, in an isolated `rl/` environment.
  Keep repository Node/npm policy and reuse its TypeScript tooling. Do not alter
  the system Python installation. Document CPU and CUDA dependency installation.

## 3. TypeScript–Python communication

Training architecture:

```text
Python: PPO + shared policy + rollout storage
  ↕ length-prefixed MessagePack over stdin/stdout
Node worker per concurrent game: real TS engine + RL adapter
```

Python starts long-lived Node child processes. Each process owns one independent
game and its mutable map. Send step requests to all workers before waiting for the
responses, so games run on multiple CPU cores. Start with one worker for a laptop
smoke test, then three for the mixed population; tune GPU-run concurrency by
measurement. More opponents do not require more Node processes within one game.

Protocol decisions:

- Commands: `hello`, `reset`, `step`, `close`. Every response includes request,
  environment, episode, and tick identifiers. Handshake checks protocol,
  observation, action, and engine compatibility versions.
- A four-byte big-endian length prefixes each MessagePack message. Large numeric
  arrays use binary buffers with explicit dtype, shape, and little-endian tensor
  bytes. Validate lengths, dimensions, finite values, and a frame-size limit.
- Reserve stdout exclusively for frames; route all engine logs to stderr. Handle
  partial reads, worker crashes, timeouts, cancellation, and child-process cleanup.
- Send public world observations once per game decision, not one full copy per
  player. Actor-private fields and legal masks are separate. Store world frames
  once in the rollout buffer and reference them from player trajectories. Include
  visible nations and tribes as entities/targets; never truncate them to a
  100-player cap. Serialize sparse relationships compactly and bound world-frame
  minibatches/activation memory for the full 423/463/513-entity starting presets.
- Send game facts needed for rewards; calculate reward formulas only in Python.
  Preserve exact large gold amounts as decimal strings when exact values are
  needed across the boundary; do not silently cast arbitrary `bigint` to Number.
- All policy players choose from the same decision-boundary state. Apply their
  actions in a documented deterministic order, then advance the configured number
  of ticks. Nations and tribes run at their native engine cadence during those ticks;
  the one-action-per-second limit applies to policy players. Normal engine
  validation resolves conflicts. Never step the world once per policy player.

Use maintained [TS/Node/browser](https://github.com/msgpack/msgpack-javascript)
and [Python](https://msgpack-python.readthedocs.io/en/latest/) MessagePack libraries.
Keep a small shared contract plus golden fixtures, rather than adding gRPC, HTTP
requests per action, or a Python reimplementation of the game's zbin protocol.

## 4. Real-engine environment and automated construction

Use `createGameRunner` and the real intent/execution system. Adapt the existing
Node map-loader pattern from `tests/perf/fullgame` into production-appropriate
shared code; runtime code must not import the test harness.

Add a narrow headless runner option to skip visual name placement and rendering
serialization without changing simulation behavior. Continue draining transient
updates correctly. Confirm state/hash parity against the normal runner.

Build one platform-neutral RL game adapter for Node and the browser worker. It
owns seeded spawn setup, observations, action masks/conversion, and the builder
schedule. Keep game balance rules in the existing engine.

Run the deterministic builder once per simulated second, with independent seeded
state per player, in the same documented tick order in training and local play.
Enabled cities, factories, defenses, silos, SAMs, and their normal economic effects
remain real engine executions. Builder decisions do not consume the policy's
action. Expose an optional builder behavior-difficulty setting, leaving the normal
game's default behavior unchanged. Keep resource-saving behavior compatible with
buying nukes; do not spend every available gold unit automatically.

The engine represents nuke purchases as `build_unit` intents, but the RL interface
must distinguish **weapon launches**, which the policy controls, from **structure
construction**, which it does not. Allow only the approved nuclear unit types
through the launch adapter. Built-in nation opponents retain their own building,
attack, diplomacy, and nuke logic; they do not run the learner's builder twice.

Use the engine's winner determination. A last surviving player does not imply an
immediate win if the normal rules still require expansion. A player death ends
that player's trajectory; other players continue. A test time limit is a
truncation, not a fabricated victory or loss.

## 5. Inputs, network, and outputs

### Inputs

Use structured state, not screenshots:

| Input | Representation |
| --- | --- |
| Player facts | Variable-length player tokens: territory, available troops, committed troops, capacity, actual/peak growth, visible economy/building statistics, alive status, visible player type |
| Relationships | Exact borders, directed attacks and troop commitments, visible alliances and expiry times |
| Geography | Initially a 64×64 occupancy-fraction grid for each player's territory, plus neutral land/fallout; shared small spatial encoder |
| Nuclear state | Visible structure positions/types/levels, silo readiness, SAM coverage, in-flight missiles and visible targets; legal weapon/target-player masks |
| Acting player's context | Incoming/outgoing alliance requests, betrayal status/timers, action availability, own private state where applicable |
| Global context | Elapsed simulated time, neutral-land share, alive counts by player type, initial policy/nation/tribe counts, public difficulty setting |

Audit visibility field by field: shared world tokens contain only information
available to all players. Private information stays in the relevant actor's
context. Do not expose opponents' queued decisions, internal bot state, or RNG.

Maintain spatial aggregates incrementally where practical. Coarse grids supply
strategy/shape information; exact engine borders determine attack legality, so a
thin border cannot disappear because of downsampling. Use log-scaled resource
inputs and ratios rather than raw million-sized numbers. Include missing/hidden
feature indicators. Normalization used for observations need not match rewards.

### First model

Start with **roughly 1 million parameters**, measuring and reporting the actual
count once instantiated:

- Shared player/spatial encoder; two attention layers, width 128, four heads.
- Relationship information incorporated into attention/target features.
- An actor-specific GRU with hidden size 256 for temporal context.
- Separate action, player/attack target, attack-fraction, and value heads. Nuclear
  launches reuse the target-player scorer; no strike-tile head. Re-measure
  parameter count after adding the nuclear observation features.

Encode the public world once per policy version per game decision, then batch the
actor-specific heads. A target's score depends on the acting player's context and
contextualized world tokens: scoring weak player A can therefore account for
strong neighbor B. Private actor context never enters another player's inputs.

No player-ID embeddings or fixed 200-player output layer. Padding exists only for
batching and is masked. The same weights accept 10, 50, 100, or other player counts
within memory limits. This is architectural support, not a guarantee of equally
good strategy at unseen lobby sizes; include held-out 25/75/125-player evaluations.

One controlled player needs one actor decision each simulated second. For 100
self-play participants, batch 100 decisions with shared weights and separate
recurrent states; do not create 100 independently trained model copies. Additional
nations and tribes use their existing TS controllers and consume simulation CPU,
not policy inferences. Include them in observations and in whichever attack or
diplomacy target sets the engine permits. Benchmark with tribes enabled: the world
encoder sees many more entities than the number of policy-controlled players.

### Outputs

Use a masked, conditional action distribution:

1. Type: wait, attack, cancel attack, request/accept alliance, reject alliance,
   renew alliance, break alliance, launch atom bomb, launch hydrogen bomb, or
   launch MIRV.
2. Target when applicable: any legal adjacent player or neutral land for attack;
   appropriate player for diplomacy; an active own attack for cancellation;
   a legal target player for a nuke, including non-adjacent targets.
3. Attack fraction when applicable: initially 5%, 10%, 20%, 35%, 50%, 75%, 100%.

Nuclear decision flow:

```text
PPO: launch(weapon_type, target_player)
  → separate StrikeTargeter: choose exact target tile
  → TS adapter: validate and submit normal nuclear launch intent
```

Define one small `StrikeTargeter` contract taking actor, target player, weapon type,
and the targeting model's required visible-state context, returning a tile or an
explicit no-valid-target result. The targeting model's implementation and training
are separate from this PPO learner; its exact input schema is a dependency to
integrate, not permission to invent another RL task here. It may use precise
spatial data without forcing PPO to predict coordinates.

Invoke it only for selected nuclear actions, before advancing that decision
boundary; batch requests when useful. Reuse the same targeter version in training,
evaluation, and local play. No aiming loss, tile log probability, or gradient enters
PPO. Store the high-level policy action/log probability plus the resolved tile and
targeter version for debugging. Validate the returned point against the intended
target, current episode/tick, and actual engine launch rules; a targeting failure
is an explicit logged no-op, not silent resampling of PPO's selected action.

Use the engine's launch legality/source-silo selection, costs, queues, cooldowns,
interception, and MIRV warhead targeting. One launch is one policy action; do not
provide unlimited stacked purchases per second. Nukes have no troop-fraction
head. A legal nuke may break an alliance through the engine; do not mask every
allied target as impossible or invent a mandatory separate break action.

The target scorer is shared over variable-length candidates. Condition the troop
fraction on the selected target. The engine's reciprocal alliance-request intent
also accepts an incoming request; the adapter exposes this clearly to the model.
`break_alliance(target)` is an explicit available choice for an active alliance;
apply the real betrayal penalties and relation changes, not a cosmetic UI change.

Mask action types that have no legal targets. Wait is always available. Store the
masks and sum the selected conditional-head log probabilities for PPO; inactive
heads contribute neither policy loss nor entropy. Convert normalized fractions
to valid integer troop counts through the engine's normal constraints.

## 6. PPO, self-play, and evaluation

Use parameter-shared, recurrent PPO with individual player returns, not a team
reward. Follow [CleanRL's PPO/recurrent implementation](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy)
as a reviewed reference, retaining attribution if code is adapted. Its Atari
implementation is not a drop-in solution for our multi-agent action space.

Initial tunable defaults in `ppo_small/constants.py`:

| Setting | Initial value |
| --- | --- |
| Learning rate / optimizer | 3e-4 / Adam, epsilon 1e-5 |
| PPO clip / update epochs | 0.2 / 4 |
| Entropy / value-loss coefficients | 0.01 / 0.5 |
| Gradient-norm cap / target KL | 0.5 / 0.02 |
| Discount per simulated second / GAE lambda at 1 s | 0.9995 / 0.99 |
| Rollout length | 128 decision boundaries per environment |
| Recurrent training sequence / burn-in | 32 steps / up to 8 preceding steps |
| Minibatch size | 32 player-sequences, reduced explicitly for laptop smoke tests |
| Population mixture | Equal training weight for 10, 50, and 100 policy slots |
| Nation presence | 80% of games with configured nations; 20% without nations |
| Tribe count | 400 in all normal training/evaluation games, configurable |
| Policy opponents | 80% all-current seats; 20% of games use half historical seats |

These are starting hypotheses, not tuned settings. Use `gamma = gamma_per_second
** decision_seconds`; apply the same discount in rewards and GAE, and scale the
lambda decay consistently when changing action frequency.

Important implementation details:

- Freeze learner weights during collection of a rollout. Update, then collect
  fresh experience. Never train PPO on arbitrary old replays, nation/tribe
  actions, or frozen-opponent actions. Store policy version, old log probability,
  value, masks, and recurrent state needed by each learner trajectory.
- Batch contiguous sequences, not shuffled individual recurrent timesteps. Reset
  hidden states on player death/new game, ignore padding, and distinguish true
  termination from rollout boundaries and time-limit truncation in GAE/bootstrap.
- Group world-frame computation across actor sequences. Recompute trainable world
  embeddings during updates; do not cache detached embeddings across PPO epochs.
- Balance losses/advantages across lobby sizes so 100-player games do not dominate
  merely because they emit more player transitions. Log the realized mixture.
- Independently of nation presence, 80% of games use current policies in all policy
  slots; 20% mix current and historical policies in equal numbers. Before history
  exists, all policy seats use the current policy. Frozen assignments stay fixed
  for a match. Bound the historical pool, initially ten snapshots; only
  current-policy seats train. Real nations continue using their own controllers.
- Shuffle seeds/spawns and policy seating. Keep scripted expansion/attack/diplomacy
  baselines for evaluation and debugging. Do not report policy-seat win rate in
  self-play as evidence of progress; games with nations/tribes are not a symmetric
  1/N baseline because their mechanics/controllers differ.

### Nation curriculum

Built-in nations provide useful competent opponents immediately, especially while
the policy is initially random. Use them alongside self-play, not as the only
opponent family: mastering fixed scripts does not establish strength against
humans or learned strategies. Retain older policy opponents to reduce forgetting;
the motivation for diverse historical opponents is illustrated by
[AlphaStar's league training](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/).
The following lightweight curriculum is a proposed experiment, not an AlphaStar
algorithm or an empirically validated schedule for this game:

| Stage | Difficulty mixture among nation-containing games |
| --- | --- |
| Bootstrap | 100% Easy |
| Intermediate | 25% Easy, 75% Medium |
| Mature | 10% Easy, 20% Medium, 70% Hard |

Difficulty is selected per game, not independently for each nation: the existing
configuration applies it globally. The engine changes both nation behavior and
their resource mechanics (Easy/Medium troop capacity is 50%/75% of the human
formula; Hard is 100%), not just tactical skill. Leave policy players on normal
human mechanics and keep their automated-builder profile fixed throughout.

Promote based on fixed, held-out benchmarks, not hours elapsed or a win threshold
borrowed from two-player games. Initial proposed gate: at least 60% wins across
40 completed games in each of two consecutive evaluation windows of a dedicated
one-learner-versus-nine-nations probe, with 400 tribes, at the current stage's main
difficulty.
Do not treat a 10/50/100-policy mixed lobby as that probe. Save thresholds, stage,
evaluation results, and mixture in checkpoints; allow an explicit logged override.
Tune the gate if it is an unhelpful bottleneck, and keep easier games in later
stages rather than replacing the entire opponent distribution overnight.

Evaluate separately at 10/50/100 policy slots with nation difficulty/count and tribe
count reported, including fixed Easy, Medium, Hard, and no-nation suites and a fixed
historical policy population, on held-out seeds with randomized seats. All these
normal evaluation suites retain tribes; label any zero-tribe ablation separately.
Report win rates with sample counts/uncertainty, final territory, survival, and
performance per lobby size. Use a fixed, versioned evaluation suite to select
`best`, not training reward. Run small evaluations frequently and larger ones
before promoting a checkpoint. Add the larger model only after this baseline is
correct and measurements identify a reason to increase capacity.

## 7. Reward and when it is computed

Compute one reward per alive learner after each decision interval, and on true
termination. Keep winning as the final objective. The **only intermediate reward
components are territory and troop-growth efficiency**; remove the previous gold,
gold-decay, maximum-troop, and current-troop reward terms.

```text
T = owned_land / initial_total_land
current_growth = engine.troopIncreaseRate(player)
peak_growth = max(engine.troopIncreaseRate(player_with_home_troops=q))
              over 0 <= q <= engine.maxTroops(player)
E = clamp(current_growth / peak_growth, 0, 1)  # zero if peak_growth <= 0

Phi(state) = 0.5 * T + 0.5 * E
reward = terminal_win_reward + shaping_scale * (gamma * Phi(next) - Phi(now))
```

`100 * E` is the requested percentage of maximum growth: the maximum achievable
**for this player's current territory, cities, capacity, and other modifiers**,
varying only home troop count. It is not troop fullness (`troops / maxTroops`), the
largest growth ever observed, or another player's growth. Units can be per engine
tick or per second as long as numerator and denominator match.

Evaluate the live engine function through a read-only hypothetical-player view;
never mutate live troops or copy the growth formula into Python. The current
curve is unimodal over valid home troops: use a bounded numeric maximum search,
including endpoints and the capacity-clamped behavior, with dense-sweep tests.
Cache by all growth-relevant state/configuration inputs and invalidate when they
change. TS sends current and peak rates as game facts; Python computes E/reward.
If troops exceed capacity and the engine returns negative growth, shaping E is
zero; keep the signed actual rate visible in observations.

Initially use equal territory/efficiency coefficients, shaping scale 0.1, and
terminal reward +1 for the engine-declared winner, 0 otherwise. Keep these in the
model's constants file. Use initial land area as the shaping denominator so
nuclear fallout does not raise T merely by shrinking eligible land; the engine's
actual victory calculation is unchanged. Gold, home/committed troops, troop
capacity, and nuclear readiness remain inputs, even though they are not separate
reward terms.

Use potential differences, not a recurring positive payment for sitting at peak
growth. High growth efficiency is not always good strategy: surviving a strong
neighbor can require holding reserves at lower efficiency. Keep shaping modest,
log both components separately, and compare against terminal-only/territory-only
shaping using wins against fixed opponents, not efficiency alone.

Set next potential to zero at a true terminal state, including an individual
player's elimination. Do not zero it at a PPO batch boundary or artificial time
limit. Use terminal observations for valid truncation bootstrap. Reward tests must
cover the discounted telescoping property, troop capacity changes, zero/negative
growth, and nuclear fallout. This is shaping to aid credit assignment, not a hard
instruction to keep growth at 100% regardless of threats.

## 8. Checkpoints and secure off-machine backup

Expose one load/save API used by training, evaluation, and local play. Follow
PyTorch's [state-dict checkpoint approach](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html),
not pickled whole model objects.

A resumable checkpoint contains:

- Model, optimizer, scheduler and optional precision-scaler state.
- RNG states, normalization state, update/agent-step counters, curriculum stage
  and evaluation history, and self-play pool metadata with referenced frozen weights.
- Resolved model/game/reward config, engine commit and map hash, observation/action
  schema versions, dependency metadata, and tracker run ID. Include the separate
  targeting model's version/config and artifact reference; make restore verify
  that dependency too, without mixing its optimizer into the PPO optimizer.

Write a temporary checkpoint, flush it, then atomically rename on the same
filesystem. Use immutable step-numbered names and a checksum manifest. Update
`latest` only after success; retain a separate evaluated `best`. Save at update
boundaries when five wall-clock minutes have elapsed, and on graceful exit.
Report checkpoint age; an exceptionally long PPO update can exceed that interval.

Resume restores learning state but **starts fresh matches in version one**.
Discard partial on-policy rollouts on interruption. Exact mid-game resume would
also require simulator, builder, and recurrent trajectory snapshots and is not
promised. Loading for CPU play needs only the policy/config/normalization subset.
Validate compatibility before loading; encode state in tensors/simple primitives
compatible with restricted `weights_only=True` loading. Reject untrusted artifacts.

Provide optional asynchronous backup to a private S3-compatible bucket, initially
Cloudflare R2. Use HTTPS, credentials scoped to a dedicated checkpoint bucket, and
immutable object names. Credentials come from environment/secret files, never Git,
CLI arguments, logs, or checkpoint metadata. Configure retention protection from
a separate administrator account, not the GPU machine: R2's
[bucket locks](https://developers.cloudflare.com/r2/buckets/bucket-locks/) protect
retained objects against deletion/overwrite. Its
[bucket-scoped object credentials](https://developers.cloudflare.com/r2/api/tokens/)
are not an assumption of write-only access; document their actual permissions.

Upload checkpoint dependencies and verify checksums before publishing the remote
manifest. Retry with bounded disk/queue usage; do not delete the last unbacked
checkpoint. Show backup failures and last successful backup age. Test download and
restore on another machine before trusting a long run. Keep limited recent local
checkpoints and configurable remote retention; do not upload rollout buffers.
Off-machine copies do not make an untrusted rental host confidential.

## 9. Live monitoring

Integrate [Weights & Biases](https://docs.wandb.ai/models/track) for a live training
dashboard. Also always write local JSONL metrics and a resolved configuration, so
the trainer works without an account or network connection. Support online,
offline-sync-later, and disabled cloud tracking modes through W&B's documented
[environment configuration](https://docs.wandb.ai/models/track/environment-variables).
Use a private project and environment-provided credentials.

Log:

- Per-population evaluation wins, territory, survival, opponent versions, nation
  count/difficulty, tribe count, curriculum stage, and curriculum-probe outcomes.
- PPO losses, KL, entropy per applicable head, clip fraction, value explained
  variance, gradient norms, and each reward component before/after weighting.
- Simulated game-seconds/second **and** learner agent-decisions/second; these are
  different metrics. Break down simulation, observations/IPC, inference, and update
  time, alongside CPU/GPU/RAM utilization and rollout memory.
- Checkpoint/update IDs, latest successful backup and its age, truncations, worker
  failures, targeting-model failures, invalid-action/no-op rates, and policy/aiming
  inference latency separately for local play.

Batch/rate-limit logging rather than logging every player every second. Keep
secrets and identifiable player data out of run metadata. W&B is a dashboard, not
the only checkpoint backup.

## 10. Play against a checkpoint locally

Reuse the existing browser game and its `LocalServer`/worker path. The browser
worker remains the sole authoritative simulator; do not run a second shadow game
in Python or Node to guess the browser's state.

1. `play` loads a chosen checkpoint on CPU, starts an inference service bound only
   to loopback, starts/connects to the development UI, and opens the local match.
2. A development-only adapter creates one human slot plus N−1 policy-controlled
   slots, configured nations, and tribes, using the same game preset, builder,
   observation extractor, and masks. Expose nation count/difficulty as explicit
   local-play overrides and report them; also expose tribe count. Use the same
   targeting model for AI nuclear strikes, while human strikes use the normal UI.
   For consistency, construction is automated for the human too in this mode;
   disable manual construction there. Human attack/diplomacy use the normal UI.
3. At each policy boundary, the browser worker supplies observations through a
   small request/response extension. A loopback WebSocket sends the same versioned
   MessagePack observation/action payloads as training, without pipe framing.
4. Python batches the AI decisions. The local turn scheduler inserts those intents
   with their assigned actor identities before advancing. Keep normal human-intent
   stamping intact; never let arbitrary browser requests impersonate players.
5. Gate the next decision boundary while inference is pending, retaining a
   responsive UI. Reject stale episode/tick responses. If inference fails, pause
   visibly with retry/exit controls rather than silently apply stale actions.

Require an ephemeral session token, allowed local Origin, bounded payloads, and
strictly validated action/player IDs. Keep credentials out of URLs sent remotely.
Disable production heartbeat, match archival, ranked/account side effects, and
analytics for this local-RL mode. Do not modify the multiplayer admin bot API: it
manages lobbies, not player gameplay. Development-only RL imports must not enlarge
or enable control paths in the normal production client.

This first local-play route requires the Python process. Browser-only model export
is a later optional step, not a prerequisite. Measure CPU p50/p95 inference at
10/50/100 players; one policy-controlled entity is much cheaper than a full local
lobby. Do not guarantee real-time 100-player performance on an unmeasured laptop.

## 11. Intended user-facing commands

These are the CLI contract to implement, not commands that work today. Examples
assume execution from the repository root after the documented dependency setup.

```bash
# Bounded laptop smoke test; artificial truncation is explicitly a test option.
uv run --project rl python -m rl train --model ppo_small --device cpu --players 10 --envs 1 --updates 2 --max-game-seconds 300 --tracking off

# Mixed policy populations plus nations and tribes; one checkpoint for all sizes.
uv run --project rl python -m rl train --model ppo_small --device cuda --players 10,50,100 --envs 12 --tracking wandb

# Restore the saved settings/state, moving devices if needed.
uv run --project rl python -m rl train --resume rl/runs/RUN/checkpoints/latest.json --device cuda

# Evaluate; play one human, nine policy players, and preset nations/tribes.
uv run --project rl python -m rl evaluate --checkpoint rl/runs/RUN/checkpoints/best.json --players 10,50,100
uv run --project rl python -m rl play --checkpoint rl/runs/RUN/checkpoints/best.json --players 10 --device cpu

# Fetch a manifest and its verified dependencies for another machine.
uv run --project rl python -m rl checkpoint download --remote s3://BUCKET/RUN/checkpoints/STEP.json --output rl/runs/restored
```

If no evaluation-selected `best` exists yet, use `latest`. A new model experiment
gets a separate run directory; incompatible architecture/reward changes are a new
run or an explicit weights-only initialization, not an unnoticed resume change.

## 12. Implementation sequence and acceptance checks

| Phase | Deliverable | Required checks before advancing |
| --- | --- | --- |
| 1. Real environment | Headless game, seeded builder/spawns, nations/tribes, nuclear targeter adapter, observations, bridge | Deterministic hashes; Node/browser parity; full population resets; launch/SAM/fallout behavior; no water creation |
| 2. Laptop PPO | Small model, recurrent PPO, rewards, local save/load, CLI smoke command | Finite losses/gradients, parameters actually update, masked actions legal, deaths/truncations correct, save/load outputs match |
| 3. Self-play and visibility | Mixed populations, nation curriculum, historical pool, fixed evaluation, W&B, backup | No nation/tribe/frozen-opponent PPO samples; stage progression/resume; per-count balancing; offline operation; secure restore |
| 4. Local opponent | CPU inference service and existing-UI local adapter | Human can complete a match against saved weights; builder/observation parity; inference failure pauses safely; no production requests/control paths |
| 5. GPU validation | Profiling, bounded GPU trial, measured scaling | CPU checkpoint resumes on CUDA; throughput/memory by player count; mixed-size training and checkpoint recovery exercised |
| 6. Experiment comparison | Larger model and/or reward ablations | Same fixed evaluation suite, multiple seeds, costs and uncertainty reported |

Additional focused tests:

- Validate the PPO update on a tiny deterministic learning task with a known
  optimum, not only by checking that its weights change. This isolates trainer
  correctness from the game's sparse outcomes and self-play instability.
- Variable N and padding/permutation invariance; changing player B can affect the
  representation used to score A. Test capacity for context, not an untrained
  network's supposed strategic intelligence.
- Neutral expansion, every adjacent opponent, minimum troops, attack cancellation,
  and the full alliance request/accept/reject/extend/break state machine, including
  real betrayal penalties and nuclear-induced breaks.
- Atom/hydrogen/MIRV legality, non-adjacent target players, purchases/cooldowns,
  SAM interception, fallout, and no created water. Do not mask nuclear launches
  merely because building construction is automated. Test the separate targeter
  with a stub, then its real implementation: legal exact tiles, consistent
  training/serving behavior, failure handling, and no spatial aiming loss in PPO.
- Normal 400-tribe spawning/controller behavior, all visible entities represented,
  legal tribe interactions, no tribe PPO samples, and memory/throughput at the full
  policy+nation+tribe population rather than just the policy-slot count.
- Growth-efficiency normalization against the live curve, dense peak-search
  validation, changes after attacks/city completion/land loss, zero/negative
  growth, fallout-safe territory normalization, bounded potentials, and terminal
  versus truncation discount handling. Confirm removal of gold/troop reward terms.
- Large gold values still survive observation/protocol conversion correctly and
  nuclear affordability checks do not round a purchase into legality.
- Incomplete frames, incompatible schemas/checkpoints, interrupted saves, failed
  uploads, bounded retries, worker restarts, and dependency-complete remote restore.
- TypeScript type checking/linting, Python formatting/type checking/tests, and
  browser integration tests for the limited existing-code hooks. Retain the
  repository's normal game tests as regression coverage.

Do not buy a long GPU run based on parameter count alone. First measure real
simulation/observation throughput: CPU game simulation may be the bottleneck.
Use the short GPU trial to estimate dollars per million learner decisions and per
completed game at each population. A few laptop games establish correctness, not
strong play; this plan makes no unmeasured promise of superhuman strength or a
fixed training budget.
