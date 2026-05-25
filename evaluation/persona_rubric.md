# Persona Alignment Scoring Rubric
#
# This rubric evaluates how well a predicted dialogue line reflects a given
# character profile within a specific conversational context. It is designed
# for LLM-as-Judge evaluation and applies to any character-grounded dialogue
# generation task.
#
# IMPORTANT CONSTRAINTS FOR THE JUDGE:
# 1. Base character_score ONLY on the character profile provided below.
#    Do NOT use any prior knowledge about the character. If the profile
#    does not mention a trait, do not reward or penalize its presence.
# 2. Base semantic_score ONLY on the dialogue context. The ground-truth
#    response is provided as a reference point to help you understand the
#    conversational moment, NOT as the only correct answer.
# 3. Score each dimension INDEPENDENTLY. A response with poor profile
#    consistency can still be contextually coherent, and vice versa.
#
# Each dimension uses a 1-5 integer scale.

## character_score — Profile Consistency

[Evaluate how consistently the predicted response reflects the character
traits DESCRIBED in the provided profile. Consider: personality, speaking
style, emotional tendencies, relationships, and behavioral patterns as
stated in the profile.

Key principle: you are measuring CONSISTENCY WITH THE PROFILE TEXT, not
authenticity to a real or fictional character. If a trait is not mentioned
in the profile, it is irrelevant to this score.]

Score 1 – No consistency.
The response is generic, flat, or interchangeable with any identity.
No described trait from the profile is discernible in the response.

Score 2 – Weak consistency.
The response reflects at most one profile trait (e.g., a slightly matching
tone), but remains largely generic. Other described traits — personality,
emotional tendencies, relationship dynamics — are absent or contradicted.

Score 3 – Moderate consistency.
The response reflects two or more described traits and adapts its tone
accordingly. The character profile is recognizably influencing the output,
but the integration is surface-level — traits appear in isolation rather
than forming a coherent characterization.

Score 4 – Strong consistency.
Multiple described traits are integrated coherently — tone, emotional
register, and interpersonal dynamics work together to produce a response
that is convincingly aligned with the profile. Minor omissions of
described traits are acceptable; no trait is contradicted.

Score 5 – Full consistency.
The response reflects the described profile comprehensively and
coherently. Personality, style, emotional state, and relational dynamics
from the profile converge naturally. The response reads as though it
could only have been produced with this specific profile as guidance.

## semantic_score — Contextual Coherence

[Evaluate whether the predicted response is a coherent and natural
continuation of the dialogue context. Do NOT consider character-style
quality here — that is captured by character_score.

The ground-truth response illustrates what kind of conversational moment
this is (e.g., humorous, emotional, confrontational, informational).
Use it to calibrate your expectations for tone and topic, but do NOT
treat it as the only valid response.

A response that takes a different but equally valid direction should
receive Score 3 or above.]

Score 1 – Incoherent.
The response is nonsensical, self-contradictory, or unrelated to the
preceding dialogue. It reads as though inserted from a different
conversation.

Score 2 – Marginally coherent.
The response connects to the scene superficially but misreads the
conversational moment — e.g., humorous when the moment is serious,
addresses a topic no one raised, or would cause confusion if spoken here.

Score 3 – Coherent continuation.
The response fits the conversational flow: it reacts to what was said,
matches the expected register, and would be accepted as a plausible next
line. It may pursue a different angle than the ground-truth.

Score 4 – Coherent with aligned intent.
A natural continuation that also addresses the same topic or expresses
a similar communicative intent as the ground-truth. The specific details
or wording differ, but the conversational function overlaps.

Score 5 – Near-equivalent continuation.
The response captures the same communicative intent, key references, and
emotional direction as the ground-truth. Wording differs but the
conversational effect is interchangeable.
