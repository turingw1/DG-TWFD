# DG-TWFD Recovery

Project-isolated recovery root:

`/temp/Zhengwei/projects/DG-TWFD`

Use this entry point after a crash:

```bash
bash /temp/Zhengwei/projects/DG-TWFD/recover/recover_all.sh
```

The script backs up any existing workspace before restoring code. Large runtime
data belongs under `/cache/Zhengwei/DG-TWFD`; this project currently keeps a
compatibility mapping to the existing runtime directory
`/cache/Zhengwei/DG-TWFD-runtime`.

Legacy evidence directories are intentionally not moved automatically:

- `/temp/Zhengwei/DG-TWFD-recovery`
- `/temp/Zhengwei/DG-TWFD-backups`

The project-isolated temp directory contains pointers and current recovery
metadata so new work can follow the v1.1 layout without breaking old artifacts.
