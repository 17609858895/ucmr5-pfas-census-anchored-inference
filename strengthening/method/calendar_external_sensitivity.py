from strengthening import paths
"""Final bounded sensitivity: frozen calendar cohort, operational f/u, R12, B100.

No candidate, tuning constant, endpoint, model or interval changes. Different
finite cohorts and budgets must not be compared by their raw errors.
"""
from pathlib import Path
import json,hashlib
from threadpoolctl import threadpool_limits
from . import external_completion_validation as ex
from . import completion_design_validation as cv
from . import range_external_validation as rv
from . import external_classical_controls as cl
P=paths.METHOD_RESULTS;D=P/'calendar_sensitivity';D.mkdir(exist_ok=True)
def main():
    live=paths.EXTERNAL_RESULTS/'calendar_aligned_systems.csv';frozen=D/'calendar_cohort_frozen.csv'
    if not frozen.exists():frozen.write_bytes(live.read_bytes())
    ex.COHORT=frozen;ex.REGIMES=['operational_learned_error'];ex.RULES=['R12'];ex.BUDGET_EQ=[100]
    ex.OUT=D/'baseline';ex.OUT.mkdir(exist_ok=True);cv.POLICIES=['uniform20','uniform40','system20','dual20'];ex.main()
    rv.D=D/'range';rv.D.mkdir(exist_ok=True);rv.main()
    cl.D=D/'classical';cl.D.mkdir(exist_ok=True);cl.main()
    (D/'final_freeze.json').write_text(json.dumps({'frozen':True,'cohort_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),'budgets_full_equivalents':[100],'rules':ex.RULES,'regimes':ex.REGIMES,'all_existing_frozen_policies_retained':True,'no_new_candidates_after_this':True,'interpretation':'A sensitivity over a different observed finite cohort and budget; raw error levels are not comparable to the primary cohort.'},indent=2),encoding='utf-8')
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
