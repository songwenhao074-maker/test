"""Preserve the exact sources matching a completed calibration's recorded hashes."""
import argparse
import json
import shutil
from train_ftmoe_end_to_end import ART, ROOT, sha


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--runs',nargs='+',required=True)
    args = parser.parse_args()
    for run in args.runs:
        root = ART/'calibrated'/run
        configuration = json.loads((root/'configuration.json').read_text())
        files = dict(configuration['evaluation_code_sha256'])
        files['calibrate_ftmoe_validation.py'] = configuration['script_sha256']
        files['artifacts/ftmoe_end_to_end/protocol_008_validation_calibration.json'] = configuration['registration_sha256']
        for relative,digest in files.items():
            source = ROOT/relative; target = root/'source_snapshot'/relative
            if target.exists(): assert sha(target)==digest; continue
            assert sha(source)==digest, f'Source no longer matches calibration: {relative}'
            target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
            assert sha(target)==digest
        print(run,'verified source snapshot',len(files))


if __name__=='__main__': main()
