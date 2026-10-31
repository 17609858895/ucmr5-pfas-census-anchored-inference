from strengthening import paths
from pathlib import Path
import sys
from threadpoolctl import threadpool_limits
from . import external_completion_validation as ex
P=paths.METHOD_RESULTS
ex.COHORT=P/'external_cohort_v2_frozen.csv'
if not ex.COHORT.exists():ex.COHORT.write_bytes((paths.EXTERNAL_RESULTS/'external_schedule_systems.csv').read_bytes())
with threadpool_limits(limits=1):
    if sys.argv[1]=='baseline':
        ex.OUT=P/'external_v2';ex.OUT.mkdir(exist_ok=True);ex.main()
    elif sys.argv[1]=='range':
        from . import range_external_validation as mod
        mod.D=P/'range_external_v2';mod.D.mkdir(exist_ok=True);mod.main()
    elif sys.argv[1]=='classical':
        from . import external_classical_controls as mod
        mod.D=P/'external_classical_v2';mod.D.mkdir(exist_ok=True);mod.main()
    else:raise ValueError('baseline / range / classical')
