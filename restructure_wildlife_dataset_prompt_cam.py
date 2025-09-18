# run once to build the folder structure PROMPT-CAM expects
import os, shutil
from pathlib import Path

def export_to_class_folders(df, root, out_root):
    out_root = Path(out_root)
    for _, r in df.iterrows():
        src = Path(root) / r['path']
        cls = f"{r['identity']}"
        dst_dir = out_root / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if not dst.exists():
            try:
                os.symlink(src.resolve(), dst)
            except Exception:
                # fallback to copy if symlink not allowed
                shutil.copy2(src, dst)

if __name__ == '__main__':
    root = "/home/richw/.richie/repos/wildlifereid-10k"
    train_out = "/tmp/wr10k_promptrun/train"
    val_out   = "/tmp/wr10k_promptrun/val"

    export_to_class_folders(df_ref, root, train_out)
    export_to_class_folders(df_qry, root, val_out)
