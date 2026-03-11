"""
app-inverse-v2: Create inverse operator for MEG/EEG source reconstruction.

Inputs : fwd.fif (from app-forward-v2),
         noise-cov.fif (from app-noise-covariance-v2),
         epochs or evoked FIF (for sensor info).
Outputs: inv.fif, optionally evoked-ave.fif.
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
    add_image_to_product,
    create_product_json,
)

setup_matplotlib_backend()
import matplotlib.pyplot as plt
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
cov_file = config.get('noise_cov') or ''
if not cov_file or not os.path.isfile(cov_file):
    add_info_to_product(report_items, f"FATAL: Noise covariance not found: '{cov_file}'.", "error")
    create_product_json(report_items)
    sys.exit(1)

try:
    noise_cov = mne.read_cov(cov_file, verbose=True)
    add_info_to_product(report_items, f"Noise covariance loaded: {noise_cov['nfree']} degrees of freedom", "info")
except Exception as e:
    add_info_to_product(report_items, f"FATAL: Could not read noise covariance: {e}", "error")
    create_product_json(report_items)
    sys.exit(1)

# == LOAD SENSOR INFO ==
# Only info is needed for make_inverse_operator — no need to load full data.
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
loose = float(config.get('loose') or 0.2)
depth = config.get('depth')
depth = float(depth) if depth not in (None, '', 'None') else 0.8

try:
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        info, fwd, noise_cov, loose=loose, depth=depth, verbose=True
    )
    add_info_to_product(
        report_items,
        f"Inverse operator: loose={loose}, depth={depth}",
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

# == OPTIONAL: COMPUTE EVOKED + PLOTS ==
compute_evoked = config.get('compute_evoked', False)
if isinstance(compute_evoked, str):
    compute_evoked = compute_evoked.lower() in ('true', '1', 'yes')

if compute_evoked and epochs_file and os.path.isfile(epochs_file):
    try:
        epochs = mne.read_epochs(epochs_file, preload=True)
        evoked = epochs.average()
        add_info_to_product(
            report_items,
            f"Averaged {len(epochs)} epochs → evoked ({evoked.nave} averages)",
            "info"
        )

        # Save evoked
        evoked_path = os.path.join('out_dir', 'evoked-ave.fif')
        evoked.save(evoked_path, overwrite=True)
        add_info_to_product(report_items, f"Saved: {evoked_path}", "info")

        # Evoked butterfly
        fig = evoked.plot(show=False, spatial_colors=True)
        fig_path = os.path.join('out_figs', 'evoked_butterfly.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        add_image_to_product(report_items, 'Evoked Response', filepath=fig_path)

        # Whitened evoked
        fig = evoked.plot_white(noise_cov, show=False)
        fig_path = os.path.join('out_figs', 'whitened_evoked.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        add_image_to_product(report_items, 'Whitened Evoked', filepath=fig_path)

        # STC time series (if method set)
        method = config.get('method') or None
        if method:
            snr     = float(config.get('snr') or 3.0)
            lambda2 = 1.0 / snr ** 2
            stc_qc  = mne.minimum_norm.apply_inverse(
                evoked, inverse_operator, lambda2, method=method, verbose=False
            )
            fig_tc, ax = plt.subplots(figsize=(10, 4))
            mean_tc = stc_qc.data.mean(axis=0)
            std_tc  = stc_qc.data.std(axis=0)
            t_ms    = stc_qc.times * 1000
            ax.plot(t_ms, mean_tc, linewidth=2, color='steelblue')
            ax.fill_between(t_ms, mean_tc - std_tc, mean_tc + std_tc, alpha=0.2, color='steelblue')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel(f'Source amplitude ({method})')
            ax.set_title(f'Source Time Course — {method}')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_path = os.path.join('out_figs', 'source_time_course.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig_tc)
            add_image_to_product(report_items, f'Source Time Course ({method})', filepath=fig_path)
            peak_vert, peak_time = stc_qc.get_peak()
            add_info_to_product(
                report_items,
                f"QC STC peak: vertex {peak_vert} at {peak_time * 1000:.1f} ms ({method})",
                "info"
            )

    except Exception as e:
        add_info_to_product(report_items, f"Could not compute evoked/plots: {e}", "warning")

# == SAVE REPORT ==
report = mne.Report(title='Inverse Operator Report')
report.save(os.path.join('out_report', 'report.html'), overwrite=True)

add_info_to_product(report_items, "Inverse operator computed successfully.", "success")
create_product_json(report_items)
print("Done.")
