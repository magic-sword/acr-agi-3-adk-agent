# Ordering and staging

Suppose a block must occupy a pad, but the only path to a switch crosses the pad. Placing the block first may prevent reaching the switch. Work backward from both conditions: open the gate via the switch, retain access while moving the block, then occupy the pad. The gate remaining open is an invariant, not another action.

If reaching the switch requires the block as a bridge, moving it away too early breaks the prerequisite. Search for a staging location or an alternate access route before accepting the plan. If neither is supported, identify the specific missing effect or route as an inquiry.

For the runtime plan, use `depends_on` only for proposed nodes. Already achieved prerequisites belong in current evidence/preconditions; never create a `done` node to assert an achievement. Each actionable node supplies `completion` and `effects`. A completion condition describes what will be observed after execution, not merely the action name.

A repair submission replaces the stored plan. Include the still-needed unaffected future work along with the repair. CLICK nodes sample the current cursor; plans for distinct click targets need fresh positioning and observation instead of a queued list of targetless clicks.
