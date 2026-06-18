from pathlib import Path
import pandas as pd
import json

ROOT = Path('/content/gdrive/MyDrive/aicup_esg_2026')
SUB = ROOT / 'submissions'

V261 = SUB / 'Sub_v26_1.csv'
V263 = SUB / 'Sub_v26_3.csv'
OUT = SUB / 'Sub_v28_1_v263_best3_v261_quality_logic.csv'
SUMMARY = SUB / '28-1PublicFieldHybridSummary.json'

REQ_COLS = ['id', 'promise_status', 'verification_timeline', 'evidence_status', 'evidence_quality']

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing: {path}')
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'{path} missing columns: {missing}')
    return df[REQ_COLS].copy()

def diff_counts(a: pd.DataFrame, b: pd.DataFrame):
    return {c: int((a[c] != b[c]).sum()) for c in REQ_COLS if c != 'id'}

def strong_logic_report(df: pd.DataFrame):
    # expected submission convention:
    # promise_status == No => timeline/evidence_status/evidence_quality should be N/A
    # evidence_status != Yes => evidence_quality should be N/A
    return {
        'promise_no_timeline_not_NA': int(((df['promise_status'] == 'No') & (df['verification_timeline'] != 'N/A')).sum()),
        'promise_no_evidence_status_not_NA': int(((df['promise_status'] == 'No') & (df['evidence_status'] != 'N/A')).sum()),
        'promise_no_evidence_quality_not_NA': int(((df['promise_status'] == 'No') & (df['evidence_quality'] != 'N/A')).sum()),
        'promise_yes_timeline_NA': int(((df['promise_status'] == 'Yes') & (df['verification_timeline'] == 'N/A')).sum()),
        'evidence_not_yes_quality_not_NA': int(((df['evidence_status'] != 'Yes') & (df['evidence_quality'] != 'N/A')).sum()),
        'evidence_yes_quality_NA': int(((df['evidence_status'] == 'Yes') & (df['evidence_quality'] == 'N/A')).sum()),
    }

v261 = read_csv(V261)
v263 = read_csv(V263)

if not (v261['id'].astype(str).values == v263['id'].astype(str).values).all():
    raise ValueError('id order mismatch between v26_1 and v26_3')

# Strategy:
# v26_3 public is better on promise_status / verification_timeline / evidence_status.
# v26_1 public is better on evidence_quality.
# So use v26_3 as base, then import v26_1 evidence_quality only when it is logically compatible
# with v26_3 evidence_status.
out = v263.copy()
mask_use_v261_quality = (out['evidence_status'] == 'Yes') & (v261['evidence_quality'] != 'N/A')
out.loc[mask_use_v261_quality, 'evidence_quality'] = v261.loc[mask_use_v261_quality, 'evidence_quality']

# Strong logic repair, fully automatic, no human inspection.
out.loc[out['promise_status'] == 'No', 'verification_timeline'] = 'N/A'
out.loc[out['promise_status'] == 'No', 'evidence_status'] = 'N/A'
out.loc[out['promise_status'] == 'No', 'evidence_quality'] = 'N/A'
out.loc[out['evidence_status'] != 'Yes', 'evidence_quality'] = 'N/A'

# If evidence_status is Yes but quality somehow N/A after repair, fall back to v26_3 quality.
mask_bad_yes_quality = (out['evidence_status'] == 'Yes') & (out['evidence_quality'] == 'N/A')
out.loc[mask_bad_yes_quality, 'evidence_quality'] = v263.loc[mask_bad_yes_quality, 'evidence_quality']
# If still N/A, use Clear as a safe non-N/A fallback.
mask_still_bad = (out['evidence_status'] == 'Yes') & (out['evidence_quality'] == 'N/A')
out.loc[mask_still_bad, 'evidence_quality'] = 'Clear'

out.to_csv(OUT, index=False)

summary = {
    'version': 'v28-1',
    'strategy': 'Use Sub_v26_3 for promise_status, verification_timeline, evidence_status; use Sub_v26_1 evidence_quality when compatible with v26_3 evidence_status; automatic strong logic repair only.',
    'input_v26_1': str(V261),
    'input_v26_3': str(V263),
    'output': str(OUT),
    'diff_vs_v26_1': diff_counts(v261, out),
    'diff_vs_v26_3': diff_counts(v263, out),
    'quality_imported_from_v26_1_count': int(mask_use_v261_quality.sum()),
    'bad_yes_quality_fixed_count': int(mask_bad_yes_quality.sum()),
    'still_bad_quality_fallback_count': int(mask_still_bad.sum()),
    'strong_logic_report': strong_logic_report(out),
    'counts': {c: out[c].value_counts().to_dict() for c in REQ_COLS if c != 'id'},
}
SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

print('[created]', OUT)
print('diff_vs_v26_1:', summary['diff_vs_v26_1'])
print('diff_vs_v26_3:', summary['diff_vs_v26_3'])
print('quality_imported_from_v26_1_count:', summary['quality_imported_from_v26_1_count'])
print('strong_logic_report:', summary['strong_logic_report'])
print('evidence_quality counts:')
print(out['evidence_quality'].value_counts())
print('[summary]', SUMMARY)
