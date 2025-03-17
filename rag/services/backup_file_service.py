from datetime import datetime
from pathlib import Path
import shutil

class BackupCreationError(Exception):
    def __init__(self, source: Path, target: Path, error: Exception):
        super().__init__(f"Failed to create backup from {source} to {target}: {str(error)}")

class BackupFileService:
    def backup(self, file_path: Path) -> None:
        if not file_path.exists():
            return

        backup_path = file_path.with_suffix(f'.{datetime.now().strftime("%Y%m%d_%H%M%S")}.bak')
        try:
            shutil.copy2(file_path, backup_path)
        except Exception as e:
            raise BackupCreationError(file_path, backup_path, e)