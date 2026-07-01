# Approach — SHL Conversational Assessment Recommender

## 1. Problem framing
A stateless agent that, over a short dialogue (≤8 turns, ≤30s/turn), recommends SHL Individual Test
Solutions. It must clarify vague asks, recommend, refine, compare named assessments, and refuse
off-topic / prompt-injection input. Scoring is primarily **Recall@10** of returned URLs against
reference shortlists, plus conversational behavior. Two facts shaped every choice: Recall@10 has **no
precision penalty** and reference shortlists are small (2–7 items), so returning more plausibly-relevant
items (up to 10) strictly helps; and the catalog + 10 reference conversations were **provided**, so I
used them as ground truth and mined them rather than scraping.

## 2. System design
```
/chat -> derive_state -> route(intent) -> {clarify | recommend/refine | compare | refuse | close}
                                              -> retrieve -> select -> guardrails -> response
```
**Stateless re-derivation.** Every call rebuilds all state from the full history — no server session
would survive the grader's independent HTTP calls. "Refine" then needs no special code: re-reading the
whole history naturally accumulates and overwrites constraints.

**ID-only structured output (anti-hallucination).** The LLM never emits a URL. The selector sees a fixed
candidate list and may only return catalog **IDs**; the service projects name/url/test_type afterward.
A fabricated link is structurally impossible — the key guarantee for a URL-graded recommender.

**Two small LLM calls/turn** on `llama-3.1-8b-instant` (Groq): a state/intent extractor and a reply
writer / leftover-slot selector. Both have deterministic fallbacks, and (see §4) the recommendation
**set** is built deterministically, so a non-deterministic or rate-limited model cannot reach an invalid
or empty state.

## 3. Retrieval — the recall bottleneck
I measured candidate recall (does the pool even *contain* the expected items?) before touching the
selector, since the selector can never recover what retrieval misses.
- **BM25** over name (×3) + description + keys + job levels — the catalog is short and matched lexically
  (Java, SQL, Docker, OPQ). A `bge-small` semantic index tested *worse* (0.42 vs 0.51 candidate recall@15)
  and added ~400MB + cold-start, so I ship BM25 only and use the LLM as the semantic layer.
- **Per-skill name-token index (biggest lever).** A long multi-skill JD dilutes BM25 so short
  single-skill tests ("SQL (New)", "Docker (New)") drop below the cut. For every concrete term the user
  types I pull both the most query-relevant and the shortest-name (most specific) catalog match.
- **Exact-named-skill priority (incl. parenthetical acronyms).** When a typed term *is* a test's full
  name or bracketed acronym ("SQL"→"SQL (New)", "AWS"→"…(AWS) Development"), that canonical test is ranked
  ahead of fuzzy variants (Oracle PL/SQL, SSIS) so it is never crowded out of the capped slots.
- **Flagship anchors.** Traces show the reference agent reliably adds flagships (OPQ32r in 7/10, Verify
  G+ in 3/10) that are lexically dissimilar to the query, so I always seed the pool with them.
- **Report families / siblings + full-term query.** A curated family map and shared-name-prefix index
  pull in report variants; the retrieval query unions raw user text with the LLM's synthesized query so a
  concrete skill is never summarized away before BM25.

These lifted candidate recall from ~0.55 to **0.86**. Remaining misses are genuinely *semantic*
(healthcare → Medical Terminology) — unreachable lexically without the embedding index that tested worse.

## 4. Selection — retrieval is the source of truth
I built and measured an LLM "ID-only selector"; on the public traces it scored **below** deterministic
assembly (≈0.53–0.58 vs 0.70 Mean Recall@10), because the model picks a narrow "representative" set while
the reference batteries include a test **per named skill**, and its derived query perturbs retrieval. So
I inverted the usual design: **deterministic retrieval owns the final set; the LLM owns the
conversation.** Each shortlist is assembled from the raw text as: carried battery → per-skill name
matches → diverse BM25 hits → flagship anchors. The LLM still writes the reply and may fill *leftover*
slots, but can never shrink the guaranteed coverage. This makes realized Recall@10 equal the
deterministic floor and **independent of free-tier latency / rate-limits** — the biggest quality win.

## 5. Multi-turn continuity
The worst failure mode was multi-turn degradation: an early turn establishes a correct battery, then a
narrow refine turn lets the stateless selector re-derive a full-but-wrong shortlist and crowd out the
technical core (one trace fell 7/7 → 2/7). Since only the final turn is graded, this tanked recall. Fix:
every recommend/compare reply embeds a `Current shortlist: …` marker; the next turn parses the latest
marker and **prepends that battery before everything else**, so each turn can only *add* to the core.
Explicit removals are honored via `test_type` exclusions. Worst trace: 0.29 → 0.57, no regressions.

## 6. Prompt design & safety
- **Separation of concerns in prompts.** The state prompt returns strict JSON (intent + constraints);
  the select prompt gets the candidate IDs + constraints and must return only IDs; the reply prompt
  writes prose. Each prompt is single-purpose with few-shot-free, schema-first instructions, which keeps
  the 8B model reliable and cheap to parse.
- **Injection defense is structural:** all history is wrapped in `<history>` tags framed as untrusted
  data; keyword heuristics are a backstop. Off-topic/injection get fixed refusal templates.
- **Grounded compare:** the compare prompt forbids prior knowledge, so durations/claims can't be invented.
- **Never 500, always in budget:** every LLM call has a timeout + deterministic fallback; `/chat` runs
  under a 26s hard cap and returns a schema-valid response on any error.

## 7. Evaluation
A replay harness feeds each reference conversation through the agent and computes Recall@10 on the final
shortlist; a probe suite checks clarify / off-topic / injection / refine / compare / turn-cap. I tracked
candidate recall (the ceiling) separately from realized recall so every miss was diagnosable as
retrieval vs selection.
- **Mean Recall@10 = 0.742** across the 10 public traces. Because the set is deterministic, this is both
  the realized score *and* the guaranteed worst case: a full replay with **every** selection call
  rate-limited scored the identical 0.742. The exact-named-skill/acronym priority lifted the mean from
  0.699 → 0.742 (C9 0.57 → 1.0) with no regressions.
- **Behavior probes: 6/6**, including under induced LLM failure. **Candidate-recall ceiling = 0.86**; the
  floor-to-ceiling gap is semantic role→skill inference a lexical pool can't reach.

## 8. Operational notes & limitations
Groq's free tier caps 70B at ~100k tokens/day, so I default to `8b-instant` (larger models swappable via
`GROQ_MODEL`). Because retrieval owns the set, token budget is a conversation-quality lever, not a recall
risk. When the deterministic floor already fills all 10 slots the agent uses a compact reply-only prompt
to avoid the per-minute limit. **Next steps:** a hybrid embedding retriever to feed the *candidate pool*
(the lever is retrieval recall, not the selector) and a learned complementary-product map for report
variants — the two remaining semantic gaps.
