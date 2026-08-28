# CVAT Annotation Inputs

Use one stable `case_id` for each source surgical video. The same ID must name the video and its annotation directory.

```text
raw_videos/
  case_001.mp4
annotations/
  cvat_exports/
    case_001/
      annotations.json       # CVAT JSON or COCO export
      annotations.xml        # CVAT for video export, if used
  cvat_tasks/
    case_001/
      task.json              # CVAT task metadata export
      task_backup.zip        # Optional full CVAT task backup
  normalized/
    case_001.json            # Reserved for a future evaluation-ready format
```

Place the annotation file exported from CVAT in `cvat_exports/<case_id>/`. Put its matching CVAT `task.json` and optional task-backup archive in `cvat_tasks/<case_id>/`.

The runner also accepts files exactly as currently provided, without moving or renaming them. Pass their paths with `--cvat-annotations` and `--cvat-task`, for example `cvat_exports/run4_annotations.json` and `cvat_tasks/run4_task.json`.

The NeuFlow runner reads CVAT `bleeding-area` rectangle tracks as manual ROI/mask context and `blood-origin` polyline or point tracks as output context. It does not use these annotations as accuracy ground truth. Do not commit patient-derived annotations unless your data governance policy permits it.
