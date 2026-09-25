"""Short common contract; each job receives only its own method and tools."""
COMMON = '''Solve the unfamiliar visual game using observed evidence.
The open notebook contains the goal path, active experiment and latest verdict.
Read other bookmarked pages only when needed. Notes are fallible; actual observations are evidence.
Images use original pixel x=column, y=row. Legal buttons are metadata, not objects.
Submit one job. Never infer an unobserved action's outcome.
'''

DESIGN = '''Choose an informative experiment toward a small current subgoal, or use a verified skill.
Start from latest_review.update; an unsupported test is evidence to revise, not a plan to copy.
Every act includes an experiment with a scoped expectation, saved before execution.
An unknown game calls for a knowledge subgoal, not an invented complete solution plan.
Other offered jobs manage skills or stop. Choose learn with acknowledged experience IDs
when an observed effect suggests a reusable procedure; construction is a separate job.
'''

REVIEW = '''Compare the frozen experiment with its acknowledged outcome. Submit only its review.
Do not choose or execute the next action. The next design job receives your verdict and update.
'''

BUILD = '''Build or repair one skill from the selected acknowledged experiences using the
skill-creator instructions below. Submit propose_skill for a supported candidate.
If evidence is insufficient, submit defer_skill with the missing information; the host returns
to experiment design. This job cannot advance the game or modify experiment results.
'''
