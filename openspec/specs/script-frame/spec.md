# script-frame Specification

## Purpose

Defines the GMUD-style structural frame a generated 武俠 RPG script must carry — factions and
stat-value threshold rules, regions, three-layer truth disclosure, chapters with hooks and
convergence points, clues, skill checks with failure fallbacks, endings, and the cost/payoff/
convergence shape every branch choice must have — plus the validators and guardrails that check
it, so a generated script is a playable RPG frame rather than a novel outline with numbers
attached.

## Requirements

### Requirement: Faction representation
The system SHALL allow a script to declare factions, each with an alignment and its relations to
other factions, so NPC and plot motivation can be grounded in a concrete power structure.

#### Scenario: Faction with relations
- **WHEN** a script declares a faction with a `stance` toward another declared faction (結盟/敵對/
  中立/附庸)
- **THEN** the reference resolves to a real faction id, and cross-reference validation reports no
  problem

#### Scenario: NPC affiliated with an unknown faction
- **WHEN** an NPC's faction reference does not match any declared faction id
- **THEN** reference validation reports the dangling reference

### Requirement: Stat-value threshold table
The system SHALL require every numeric player stat that a branch effect can change to have at
least one threshold rule declaring what that value range unlocks (a branch, an event, an NPC
attitude, or an ending), so a stat's narrative meaning is explicit rather than decorative.

#### Scenario: Stat covered by a threshold
- **WHEN** a stat is targeted by a branch's structured effect and a threshold rule's range covers
  that stat
- **THEN** stat-threshold validation reports no problem for that stat

#### Scenario: Stat modified with no threshold coverage
- **WHEN** a stat is targeted by a branch's structured effect but no declared threshold rule
  covers it
- **THEN** stat-threshold validation reports the stat as having no narrative meaning attached

#### Scenario: Overlapping threshold ranges for one stat
- **WHEN** two threshold rules for the same stat declare overlapping value ranges
- **THEN** stat-threshold validation reports the overlap

### Requirement: Region and sub-location structure
The system SHALL allow a script to declare regions, each with an unlock condition and at least
two functional sub-locations (e.g. 打聽消息, 交易, 療傷, 學習技能), and allow events to reference
which region and sub-location they take place in.

#### Scenario: Region with sub-locations
- **WHEN** a script declares a region with two or more sub-locations, each with a function label
- **THEN** the region satisfies the minimum sub-location structure

#### Scenario: Event referencing an unknown region or sub-location
- **WHEN** an event's region or sub-location reference does not match a declared id
- **THEN** reference validation reports the dangling reference

### Requirement: Three-layer truth disclosure
The system SHALL let a script separate its underlying truth into three layers — facts known from
the start (public), facts revealed progressively as chapters advance, and facts reserved for the
author that must never appear in generated scene content before their designated reveal point.

#### Scenario: Progressive reveal resolves to a real chapter
- **WHEN** a progressive-reveal fact declares the chapter or event it is revealed in
- **THEN** that id resolves to a declared chapter or event, and reveals are in non-decreasing
  chapter order

#### Scenario: Hidden truth is withheld from scene generation
- **WHEN** a scene is generated for a chapter earlier than a hidden fact's designated reveal point
- **THEN** that fact is not made available as input to that scene's generation, regardless of
  whether an automated check would also catch it appearing in the output

#### Scenario: A hidden fact leaks into scene content before its reveal point
- **WHEN** a generated event's summary or dialogue contains a hidden truth fact before that fact's
  designated reveal point
- **THEN** truth-pacing validation reports the premature disclosure

### Requirement: Chapter structure with hook and convergence
The system SHALL let a script organize events into chapters, each with an opening hook and a
convergence event that every branch path within the chapter eventually reaches, so branching
does not grow unboundedly across chapters.

#### Scenario: Chapter with a declared convergence point
- **WHEN** a chapter declares a convergence event id
- **THEN** that id resolves to a real event, and every branch that starts within the chapter can
  reach it

#### Scenario: Chapter missing a convergence point
- **WHEN** a chapter contains branching events but declares no convergence event
- **THEN** convergence validation reports the chapter as unbounded

### Requirement: Clue tracking
The system SHALL let a script declare clues, each tied to the event where it can be found and
the mystery or ability-gated path it serves, so investigative scenes carry information rather
than being pure narrative filler.

#### Scenario: Clue resolves to a real event
- **WHEN** a clue declares the event it is found in
- **THEN** that id resolves to a declared event

#### Scenario: Event with no information payoff
- **WHEN** an event unlocks no clue, item, or truth reveal
- **THEN** scene-information validation reports the event as a content-free scene

### Requirement: Skill checks with failure fallback
The system SHALL let an event declare skill checks (attribute contest, dice, probability, or item
bypass), each of which SHALL declare either a failure branch or an item-based bypass, and a
failure cost, so a failed check advances the story instead of ending it.

#### Scenario: Check with a failure branch
- **WHEN** a skill check declares a failure branch id and a failure cost
- **THEN** check-fallback validation reports no problem for that check

#### Scenario: Check with no failure route
- **WHEN** a skill check declares neither a failure branch nor an item bypass
- **THEN** check-fallback validation reports the check as a possible dead end

### Requirement: Ending declarations
The system SHALL let a script declare endings, each with the stat-value conditions and/or
required branch choices that select it, so the script has explicit, checkable win/resolution
states tied to player choices.

#### Scenario: Ending selected by a stat condition
- **WHEN** an ending declares a stat-value range condition
- **THEN** that stat resolves to a real player stat

#### Scenario: Ending selected by a required branch
- **WHEN** an ending declares a required branch id
- **THEN** that id resolves to a real branch

### Requirement: Branch choices carry cost, feedback, and delayed payoff
The system SHALL require every branch with a structured effect to declare a cost (what the player
gives up, not merely a stat delta), and, when its effect is not immediately resolved, a delayed
payoff — the chapter and description of when and how it is redeemed later — plus an eventual
convergence point, so choices are real value/resource tradeoffs rather than choices that differ
only in degree.

#### Scenario: Branch with cost and delayed payoff
- **WHEN** a branch declares a non-empty cost and, for a deferred effect, a payoff chapter that
  resolves to a chapter no earlier than the branch's own chapter
- **THEN** choice-quality validation reports no problem for that branch

#### Scenario: Branch missing a cost
- **WHEN** a branch has a structured effect but no declared cost
- **THEN** choice-quality validation reports the branch as missing a real tradeoff

#### Scenario: Two branches differ only in degree
- **WHEN** two branches within the same event have highly similar choice text, touch the same
  effect targets, and declare no distinct cost from one another
- **THEN** choice-quality validation flags the pair as a false choice (假選擇)

#### Scenario: Deferred payoff never declared
- **WHEN** a branch's structured effect is not resolved within its own event but the branch
  declares no payoff chapter or payoff description
- **THEN** delayed-payoff validation reports the branch as an undeclared deferred effect

### Requirement: Main and flavor scene balance
The system SHALL let an event declare whether it is a main scene (推進真相) or a flavor scene
(調味), and the guide's guidance that main scenes should be somewhat more numerous than flavor
scenes SHALL be checkable per script.

#### Scenario: Script with a reasonable main/flavor mix
- **WHEN** a script's main-scene count is at or above its flavor-scene count
- **THEN** scene-mix validation reports no problem

#### Scenario: Script skewed toward flavor scenes
- **WHEN** a script's flavor-scene count substantially exceeds its main-scene count
- **THEN** scene-mix validation reports the imbalance

### Requirement: Backward-compatible script data
Every field this capability adds to the script data model SHALL be optional with a default value,
so a script produced before this capability existed remains loadable and displayable without
migration.

#### Scenario: Loading a pre-existing script
- **WHEN** a previously generated script that predates this capability is loaded
- **THEN** it loads successfully with every new field resolving to its default (empty/absent),
  and none of this capability's validators report a problem caused solely by those fields being
  absent
