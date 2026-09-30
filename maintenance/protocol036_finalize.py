from pathlib import Path
import argparse
import hashlib
import json
import shutil


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')


def cmd_manifest(a):
    root = Path(a.run_root)
    rows = []
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.startswith('analysis/'):
            continue
        rows.append({'path': rel, 'sha256': sha(p), 'bytes': p.stat().st_size})
    write(a.output, {'protocol':'036','archived_before_analysis':True,'file_count':len(rows),'files':rows})


def copy_if(src, dst):
    src = Path(src); dst = Path(dst)
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def cmd_compact(a):
    root = Path(a.run_root); dest = Path(a.dest); dest.mkdir(parents=True, exist_ok=True)
    top = ['budget_ledger.json','budget_ledger_events.jsonl','fixture_report.json','implementation_manifest.json','source_provenance.json','raw_artifact_manifest.json','scientific_execution_complete.json','pip_freeze.txt']
    for name in top: copy_if(root/name, dest/name)
    for name in ['comparison.json','cost_profile.json','status.json']:
        copy_if(root/'analysis'/name, dest/name)
    for seed in (3601,3602,3603):
        base=root/'stage_B'/f'seed{seed}'
        for rel in ['C_ref/summary.json','C_ref/causal_access_audit.json','D_cal/summary.json','D_cal/causality_audit.json','D_cal/branch_isolation_audit.json','D_lin/summary.json','D_lin/causality_audit.json','D_lin/branch_isolation_audit.json','D_corr/summary.json','D_corr/causality_audit.json','D_corr/branch_isolation_audit.json']:
            copy_if(base/rel,dest/f'seed{seed}'/rel)
        copy_if(root/'inputs'/f'seed{seed}'/'input_audit.json',dest/f'seed{seed}'/'input_audit.json')
        copy_if(root/'inputs'/f'seed{seed}'/'input_lock.json',dest/f'seed{seed}'/'input_lock.json')
        copy_if(root/'inputs'/f'seed{seed}'/'manifest.json',dest/f'seed{seed}'/'stream_manifest.json')
    for arm in ('D_cal','D_lin'):
        copy_if(root/'stage_A_seed700'/arm/'summary.json',dest/'stage_A'/arm/'summary.json')
    index=[]
    for p in sorted(dest.rglob('*')):
        if p.is_file(): index.append({'path':p.relative_to(dest).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
    write(dest/'compact_manifest.json',{'protocol':'036','run_id':str(a.run_id),'files':index})


def cmd_pointers(a):
    status=json.loads(Path(a.status).read_text(encoding='utf8'))
    label=status['registered_result_label']; rid=str(a.run_id); compact=str(a.compact)
    Path('AGENTS.md').write_text(
        f'# Active Experiment Directive\n\nProtocol-036 scientific execution is complete in run {rid}. Registered result label: {label}.\n\nPrimary handoff: `docs/PROTOCOL036_RESULTS.md`. Compact evidence: `{compact}/`.\n\nExactly 14 registered training sequences and 3 registered new streams were consumed. Do not rerun Protocol-035 or Protocol-036 budgets. No automatic follow-up experiment is authorized.\n',encoding='utf8')
    Path('NEXT_EXPERIMENT_LATEST.md').write_text(
        f'# 下一实验交接\n\nProtocol-036 已完成，run `{rid}`，登记结果标签 **{label}**。\n\n先读取 `docs/PROTOCOL036_RESULTS.md`、`{compact}/comparison.json`、`{compact}/status.json` 和三条流的输入/因果审计。036 的 14 条训练序列与 3 条新流预算已关闭，不得重跑或按结果追加种子/扫参。下一步仅在新的预登记指示发布后执行。\n',encoding='utf8')
    Path('PROJECT_CONTEXT_LATEST.md').write_text(
        f'# Project Context Latest\n\nCurrent terminal experiment: Protocol-036, run `{rid}`. Registered result label: **{label}**. Scientific status: `{status["scientific_status"]}`. Cross-stream gain signal: `{status["cross_stream_gain_signal"]}`. D_corr over D_cal incremental signal: `{status["D_corr_over_D_cal_incremental_signal"]}`. D_corr over D_lin incremental signal: `{status["D_corr_over_D_lin_incremental_signal"]}`. Operating-point risk: `{status["operating_point_risk"]}`.\n\nRead `docs/PROTOCOL036_RESULTS.md` and `{compact}/` for current evidence. Historical Protocol-033/034/035 registrations and evidence remain immutable.\n',encoding='utf8')


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('manifest'); p.add_argument('--run-root',required=True); p.add_argument('--output',required=True); p.set_defaults(fn=cmd_manifest)
    p=sub.add_parser('compact'); p.add_argument('--run-root',required=True); p.add_argument('--dest',required=True); p.add_argument('--run-id',required=True); p.set_defaults(fn=cmd_compact)
    p=sub.add_parser('pointers'); p.add_argument('--status',required=True); p.add_argument('--run-id',required=True); p.add_argument('--compact',required=True); p.set_defaults(fn=cmd_pointers)
    a=ap.parse_args(); a.fn(a)


if __name__=='__main__': main()
