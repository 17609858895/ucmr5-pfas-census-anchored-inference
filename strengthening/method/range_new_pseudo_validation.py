from strengthening import paths
"""New10–14 pseudo-populations for the range-bound completion candidate.

Old0–9 pseudo-results are development information. No old holdout is relabeled.
The candidate and cost-matched Neyman comparator are fixed before this run.
"""
from pathlib import Path
import json,hashlib
from . import completion_design_validation as cv
from threadpoolctl import threadpool_limits
P=paths.METHOD_RESULTS;D=P/'range_newseeds10_14';D.mkdir(exist_ok=True)
spec={'development_pseudo_seeds_consumed':list(range(10)),'new_validation_pseudo_seeds':list(range(10,15)),'policies':['uniform20','uniform40','neyman_cost20','range_cost20'],'new_policy_floor':.1,'new_policy_cap':1.,'range_score':'max(1,P/mean(P))/first_stage_score','pilot_incremental_cost_fraction':.2,'cost_predictor':'historical source-group incremental/fullcost ratio multiplied by declared full-schedule cost; target missed labels never used','outcome_inspection_history':'Candidate motivated by old pseudo-results and Michigan. Prior policies had already been computed on Washington but Washington policy performance was not inspected before this candidate was fixed; do not call Washington blind/preregistered.','candidate_type':'Minimax importance-range adaptation for the correction component, not a new global optimum or coverage guarantee.'}
(D/'candidate_prespecification.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
cv.OUT=D;cv.SEEDS=list(range(10,15));cv.POLICIES=spec['policies']
with threadpool_limits(limits=1):cv.main()
