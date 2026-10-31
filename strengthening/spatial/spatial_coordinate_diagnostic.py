from strengthening import paths
"""Exploratory administrative-identity ablation, fixed before new-state outcomes.

This diagnostic follows the first cold-region evaluation and is not a blind
confirmatory comparison: it tests whether unseen state/region identities help
explain the observed cold-region calibration failure. Original results remain.
"""
from pathlib import Path
import argparse,json
from . import spatial_transfer as st
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--frame',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    args=a.parse_args()
    st.SETS={'coordinates':st.ATTR+['lat','lon'],'coordinates_neighbours':st.ATTR+['lat','lon']+st.NB}
    st.run(args.frame,args.out)
    (args.out/'diagnostic_status.json').write_text(json.dumps({'status':'Exploratorydiagnostic after initial coldregion results; no extra external outcomes used for modelchoice.','question':'Do unseen administrativeidentities cause part of calibration failure?','change':'OmitstateandEPAregion IDs; retainrawlatitude/longitude and same attributes, fixedhyperparameters, matchedseeds and spatialexclusion.'},indent=2),encoding='utf-8')
