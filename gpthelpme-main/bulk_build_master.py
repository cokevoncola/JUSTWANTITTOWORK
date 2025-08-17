# src/bulk_build_master.py
from __future__ import annotations
import argparse, gzip, io, os, sys, subprocess
from pathlib import Path
from typing import Iterable, List

def natural_sort_key(p: Path):
    import re
    s = str(p)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def find_files(raw_dir: Path, pattern: str) -> List[Path]:
    files = sorted(raw_dir.glob(pattern), key=natural_sort_key)
    return [p for p in files if p.is_file()]

def concat_csv_like(inputs: List[Path], out_path: Path) -> None:
    """
    Concatenate CSV / CSV.GZ files into a gzipped CSV at out_path.
    Writes header once (from first file), skips headers of subsequent files.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with gzip.open(out_path, "wt", newline="") as w:
        for f in inputs:
            is_gz = f.suffix == ".gz"
            opener = (lambda p: gzip.open(p, "rt", encoding="utf-8", errors="ignore")) if is_gz \
                     else (lambda p: open(p, "rt", encoding="utf-8", errors="ignore"))
            with opener(f) as r:
                for i, line in enumerate(r):
                    if i == 0:
                        if not wrote_header:
                            w.write(line)
                            wrote_header = True
                        # else: skip this header
                    else:
                        w.write(line)

def rebuild_profiles(master_path: Path, out_dir: Path | None) -> None:
    """
    Try to import and call src.build_profiles.main(master, out).
    Fallback to subprocess if import fails for any reason.
    """
    print("🔧 Rebuilding profiles from master …")
    try:
        from src.build_profiles import main as build_profiles_main  # type: ignore
        build_profiles_main(str(master_path), str(out_dir) if out_dir else None)
        return
    except Exception as e:
        print(f"   Direct import path failed: {type(e).__name__} - {e}")
        print("   Falling back to CLI execution of src.build_profiles …")
        py = sys.executable or "python"
        cmd = [py, "-m", "src.build_profiles", "--master", str(master_path)]
        if out_dir:
            cmd += ["--out", str(out_dir)]
        print("   (running)", " ".join(cmd))
        subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="Folder containing raw CSV/CSV.GZ files")
    ap.add_argument("--pattern", default="*.csv*", help="Glob pattern (default: *.csv*)")
    ap.add_argument("--out", default="data/statcast_master.csv.gz", help="Output gzipped master path")
    ap.add_argument("--rebuild_profiles", action="store_true", help="Rebuild profiles after writing master")
    ap.add_argument("--profiles_out", default=None, help="Optional output dir for profiles (defaults to models/ or config)")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_path = Path(args.out).resolve()
    files = find_files(raw_dir, args.pattern)

    if not files:
        print(f"No files found in {raw_dir} matching {args.pattern}")
        sys.exit(1)

    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  - {f.name}")

    print(f"↻ Reading {files[0].name} …")
    for f in files[1:]:
        print(f"↻ Reading {f.name} …")

    print(f"🗜️  Gzipping to {out_path} …")
    concat_csv_like(files, out_path)
    print(f"✅ Master written: {out_path}  (from {len(files)} file(s))")

    if args.rebuild_profiles:
        # pick a default output dir for profiles
        if args.profiles_out:
            prof_out = Path(args.profiles_out).resolve()
        else:
            try:
                from src import config as CFG  # type: ignore
                prof_out = Path(getattr(CFG, "MODELS_DIR", "models")).resolve()
            except Exception:
                prof_out = Path("models").resolve()
        rebuild_profiles(out_path, prof_out)

if __name__ == "__main__":
    main()
