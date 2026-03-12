"""
app-inverse-v2: Create inverse operator for MEG/EEG source reconstruction.

Authors : Guiomar Niso (guiomar.niso@gmail.com)
          Antonio Caulín (antoniocaulinatienzar@gmail.com) https://github.com/AntonioCauAt
Based on: https://github.com/guiomar/app-inverse-operator

Inputs : fwd.fif (from app-forward-v2),
         noise-cov.fif (from app-noise-covariance-v2),
         epochs or evoked FIF (for sensor info only).
Outputs: inv.fif
"""

import os
import sys

app_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(app_dir)
for search_path in [app_dir, parent_dir]:
    if os.path.isdir(os.path.join(search_path, 'brainlife_utils')):
        sys.path.insert(0, search_path)
        break

from brainlife_utils import (
    setup_matplotlib_backend,
    load_config,
    ensure_output_dirs,
    add_info_to_product,
    create_product_json,
)

setup_matplotlib_backend()
import mne

# == SETUP ==
ensure_output_dirs('out_dir', 'out_figs', 'out_report')
report_items = []

# == LOAD CONFIG ==
config = load_config()

# == LOAD FORWARD SOLUTION ==
fwd_file = config.get('forward') or ''
if not fwd_file or not os.path.isfile(fwd_file):
    add_info_to_product(report_items, f"FATAL: Forward solution not found: '{fwd_file}'.", "error")
    create_product_json(report_items)
    sys.exit(1)

try:
    fwd = mne.read_forward_solution(fwd_file, verbose=True)
    add_info_to_product(report_items, f"Forward: {fwd['nsource']} sources", "info")
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not read forward solution: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == LOAD NOISE COVARIANCE ==
cov_file = config.get('noise_cov') or config.get('cov') or ''
if not cov_file or not os.path.isfile(cov_file):
    add_info_to_product(report_items, f"FATAL: Noise covariance not found: '{cov_file}'.", "error")
    create_product_json(report_items)
    sys.exit(1)

try:
    noise_cov = mne.read_cov(cov_file, verbose=True)
    add_info_to_product(report_items, f"Noise covariance: {noise_cov['nfree']} degrees of freedom", "info")
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not read noise covariance: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == LOAD SENSOR INFO ==
epochs_file = config.get('epo') or config.get('epochs') or None
evoked_file = config.get('evoked') or None

info = None
try:
    if epochs_file and os.path.isfile(epochs_file):
        info = mne.read_epochs(epochs_file, preload=False).info
        add_info_to_product(report_items, f"Sensor info from epochs: {len(info['ch_names'])} channels", "info")
    elif evoked_file and os.path.isfile(evoked_file):
        info = mne.read_evokeds(evoked_file)[0].info
        add_info_to_product(report_items, f"Sensor info from evoked: {len(info['ch_names'])} channels", "info")
    else:
        add_info_to_product(report_items, "FATAL: No sensor data found. Set 'epo' or 'evoked' in config.", "error")
        create_product_json(report_items)
        sys.exit(1)
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not load sensor info: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == MAKE INVERSE OPERATOR ==
_loose = config.get('loose')
loose  = float(_loose) if _loose not in (None, '', 'None', 'auto') else 'auto'

_depth = config.get('depth')
if _depth in (None, ''):
    depth = 0.8                          # default
elif str(_depth).lower() == 'none':
    depth = None                         # explicit: no depth weighting
else:
    depth = float(_depth)

_rank = config.get('rank')
if _rank in (None, '', 'None', 'auto'):
    rank = None
elif _rank == 'info':
    rank = 'info'
elif _rank == 'full':
    rank = 'full'
else:
    rank = None

try:
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        info, fwd, noise_cov, loose=loose, depth=depth, rank=rank, verbose=True
    )
    add_info_to_product(
        report_items,
        f"Inverse operator: loose={loose}, depth={depth}, rank={rank}",
        "info"
    )
except Exception as e:
    add_info_to_product(report_items, f"FATAL: make_inverse_operator failed: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == SAVE INVERSE OPERATOR ==
inv_path = os.path.join('out_dir', 'inv.fif')
try:
    mne.minimum_norm.write_inverse_operator(inv_path, inverse_operator, overwrite=True)
    add_info_to_product(report_items, f"Saved: {inv_path}", "info")
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not save inverse operator: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == SAVE REPORT ==
report = mne.Report(title='Inverse Operator Report')
report.save(os.path.join('out_report', 'report.html'), overwrite=True)

add_info_to_product(report_items, "Inverse operator computed successfully.", "success")
create_product_json(report_items)
print("Done.")
