# PHASE-Tree State-Update Human Evaluation Guidelines

## Evaluation Task

You are an independent human evaluator. Determine whether an update from `old_value` to `new_value` for a target character's state field is supported by dialogue from the target episode and preceding episodes.

Focus on the information introduced, removed, or strengthened by the update. You do not need to re-establish unchanged information carried over from `old_value`.

## Available Information

Each evaluation item contains:

- `update_id`
- The work and target character
- The target `episode`
- The updated `field`
- The state before the update, `old_value`
- The state after the update, `new_value`
- Relevant dialogue from the target episode and preceding episodes
- Extracted evidence summaries, `evidence`
- The update model's explanation, `reasoning`

The `reasoning` field may help clarify the intended update, but it is not evidence that the update is correct.

## How to Read the Episode Dialogue

1. Use only information from the target episode and earlier episodes. Do not use events from later episodes.
2. Give evidence priority in the following order:
   - Explicit statements or actions by the target character
   - Direct statements about the target character made by other characters
   - Consistent behavioral patterns across multiple scenes
   - Narrative summaries or indirect inferences
3. Do not use external encyclopedias, character profiles, actor information, or your own prior knowledge of the work.
4. Evidence must concern the target character and the target field. Another character's behavior cannot directly support an update to the target character.
5. A single situation-specific behavior is generally insufficient to establish a stable personality change unless it represents a clear and significant turning point.
6. If earlier and current evidence conflict, determine whether `new_value` accurately represents the character's latest state as of the target episode.

## Evaluation Procedure

### Step 1: Identify the Actual Change

Compare `old_value` with `new_value` and identify:

- Newly added facts or attributes
- Information that was removed or replaced
- Changes in degree, such as from "occasionally" to "increasingly"
- Changes in state, such as from friend to romantic partner or from employee to manager

Base the main judgment only on these changes.

### Step 2: Examine the Evidence

For each substantive change, determine whether it:

- Is supported by direct dialogue or narrative action
- Is supported only through indirect inference
- Has no supporting evidence
- Contradicts the available evidence
- Overgeneralizes a temporary state into a long-term attribute

### Step 3: Select a Label

#### `supported`

Choose `supported` when:

- Every major change has direct or strong, consistent evidence
- The update introduces no important unsupported information
- The update does not conflict with information available through the target episode
- Abstract updates, such as personality changes, are supported by repeated behavior or a clear turning point

Minor wording differences that do not change the meaning are acceptable.

#### `partial`

Choose `partial` when any of the following applies:

- The central direction of the update is correct, but some details are unverified
- The evidence is relevant but indirect
- Only some of several substantive changes are adequately supported
- The wording is too strong or overgeneralized
- A one-time or situation-specific behavior is presented as a stable characteristic
- The evidence is mixed, but the update still has a reasonable basis

#### `unsupported`

Choose `unsupported` when any of the following applies:

- The central change has no relevant evidence
- The update clearly contradicts the dialogue or narrative
- The evidence concerns another character and does not support the target character
- The update depends on events occurring after the target episode
- The claim appears only in `reasoning` and is not supported by dialogue evidence
- Most of the update consists of incorrect facts or unsupported inference

## Field-Specific Requirements

- `occupation`, `demographics`, and `relationships`: Prefer explicit factual evidence.
- `personality` and `behavioral_tendencies`: Require repeated behavior or a clear, significant turning point.
- `speaking_style`: Judge from the target character's actual speech, not solely from narrative events.
- `hobbies` and `skills`: A single action does not necessarily establish a lasting interest or stable ability.

## Output Format

Return:

```json
{
  "update_id": "<ID>",
  "label": "supported | partial | unsupported",
  "note": "In one or two sentences, identify which changes are supported and describe the main limitation or contradiction."
}
```
