import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional,Any, Union , List
from datetime import datetime

from app.config import settings

logger = logging.getLogger(__name__)

class FileSystemTool:
    def __init__(self):
        
        self.workspace  =settings.WORKSPACE_DIR
        self.allowed_read_extentions = settings.ALLOWED_READ_EXTENSIONS
        self.allowed_write_extentions = settings.ALLOWED_WRITE_EXTENSIONS
        self.max_read_size  = settings.MAX_READ_SIZE
        self.max_write_size  = settings.MAX_WRITE_SIZE
        
        self.inputs_dir = self.workspace / "inputs"
        self.outputs_dir = self.workspace / "outputs"
        self.temp_dir = self.workspace / "temp"
        
        self._ensure_directories()
        
        logger.info(f"✅ FileSystemTool initialized. Workspace: {self.workspace}")

    def _ensure_directories(self) -> None:
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path : Union[str, Path])->Path:
        
        if isinstance(path, str):
            path = Path(path)
        
        if path.is_absolute():
            resolved = path.resolve(path)
            workspace_resolved = self.workspace.resolve()

            if not str(resolved).startswith(str(workspace_resolved)):
                raise ValueError(
                    f"Access denied: Path '{path}' is outside the workspace.\n"
                    f"Agent can only access: {self.workspace}"
                )
            return resolved
        
        else:
            resolved = (self.workspace / path).resolve()
            workspace_resolved = self.workspace.resolve()
            
            # Check: Is resolved path inside workspace?
            if not str(resolved).startswith(str(workspace_resolved)):
                raise ValueError(
                    f" Access denied: Path '{path}' resolves to '{resolved}',\n"
                    f"   which is outside the workspace.\n"
                    f"   Agent can only access: {self.workspace}"
                )
            
            return resolved
    
    def _validate_extentions(self, path : Path, allowed_extentions: List[str] ) -> None:
        
        ext = path.suffix.lower()

        if not ext :
            return
        if ext not in allowed_extentions : 
            raise ValueError(
                f"❌ File extension '{ext}' is not allowed.\n"
                f"   Allowed: {', '.join(allowed_extentions)}"
            )
        
    def _validate_size(self, size : int, max_size : int , operation:  str = "read") ->None:
        
        if size > max_size:
            print("CONTENT IS VERT LONG , CAN'T FIT IN THE FILE")
            size_mb = size(1024*1024)
            max_mb = size(1024*1024)
            raise ValueError(
                f"❌ File size ({size_mb:.2f}MB) exceeds maximum {operation} size ({max_mb:.2f}MB)"
            )
    
    def _log_opertion(self, operation: str, path :Path, success: bool, details: Optional[Dict[str, Any]] = None) ->None:
        
        if success:
            logger.info(f"✅ File {operation} successful: {path}")
        else:
            logger.warning(f"❌ File {operation} failed: {path}")
        if details:
            logger.debug(f"   Details: {details}")

    def read_file(self, path :Union[str, Path], encoding : str  = "utf-8") -> Dict[str,Any]:
        
        try:
            resolve_path = self._resolve_path(path)  
            
            if not resolve_path.exists():
                raise ValueError(f"filr not found {resolve_path}")
            
            if not resolve_path.is_file():
                raise ValueError(f"path is not a file {resolve_path}")
            
            self._validate_extentions(resolve_path,self.allowed_read_extentions)
            
            file_size = resolve_path.stat().st_size
            self._validate_size(file_size,self.max_read_size, "read")
            
            content = resolve_path.read_text(encoding=encoding)
            
            self._log_opertion("read", resolve_path,True, {"size": file_size , "encoding" : encoding})
            
            return {
                "success" : True,
                "content" : content,
                "path" : str(resolve_path),
                "size" : file_size,
                "encoding" : encoding
            }
        except Exception as e:
            self._log_opertion("read" , path , False , {"error" : str(e)})
            return {
                "success": False,
                "error": str(e),
                "path": str(path),
            }
    def write_file(self, path : Union[str, Path], content :str, encoding:str = "utf-8",overwrite:bool = True)->Dict[str, Any]:
        
        print("writing the file \n")
        try:
            print(f"content = {content}\n")
            resolved_path = self._resolve_path(path)
            self._validate_extentions(resolved_path, self.allowed_write_extentions)
            
            if  resolved_path.exists() and not overwrite:
                print("cant overwrite this file")
                raise ValueError(
                    f"File {resolved_path} already exists. "
                    f"Set overwrite=True to replace it."
                )
            
            content_bytes = content.encode(encoding)
            content_size = len(content_bytes)
            self._validate_size(content_size,self.max_write_size , "write")
            
            resolved_path.write_text(content, encoding=encoding)

            self._log_opertion("write", resolved_path, True, {
                "size": content_size,
                "encoding": encoding,
                "overwrite": overwrite,
            })
            print(" done writing the file  \n")
            return {
                "success": True,
                "path": str(resolved_path),
                "size": content_size,
                "encoding": encoding,
            }
        except Exception as e:
            self._log_opertion("write", path, False, {"error": str(e)})
            return {
                "success": False,
                "error": str(e),
                "path": str(path),
            }
    
    def list_files(self,path : Union[str, Path], recursive :bool = False , pattern :Optional[str] = None)->Dict[str, Any]:
        
        try:
            resolved_path = self._resolve_path(path)
            
            if not resolved_path.exists():
                raise ValueError(f"Directory not found: {resolved_path}")
            
            if not resolved_path.is_dir():
                raise ValueError(f"Path is not a directory: {resolved_path}")

            files = []
            directories = []
            
            if recursive:
                for root, dirs, filenames in os.walk(resolved_path):
                    rel_root = Path(root).relative_to(self.workspace)
                    
                    for dir_name in dirs:
                        directories.append(str(rel_root/dir_name))
                    for filename in filenames:
                        file_path = rel_root/filename
                        if pattern and not Path(filename).match(pattern):
                            continue
                        files.append(str(file_path))
            else:
                workspace_resolved = self.workspace.resolve()
                
                for item in resolved_path.iterdir():
                    rel_path = item.relative_to(workspace_resolved)
                    if item.is_file():
                        if pattern and not item.match(pattern):
                            continue
                        files.append(str(rel_path))
                    elif item.is_dir():
                        directories.append(str(rel_path))
            self._log_opertion("list", resolved_path, True, {
                "files_count": len(files),
                "directories_count": len(directories),
                "recursive": recursive,
                "pattern": pattern,
            })
            return {
                "success": True,
                "path": str(resolved_path),
                "files": files,
                "directories": directories,
                "count": len(files) + len(directories),
            }
        except Exception as e:
            self._log_opertion("list", path, False, {"error": str(e)})
            
            return {
                "success": False,
                "error": str(e),
                "path": str(path),
            }
    def delete_file(self, path :Union[str, Path]) -> Dict[str, Any]:
        
        try:
            resolved_path = self._resolve_path(path)
            
            if not resolved_path.exists():
                raise ValueError(f"File not found {resolved_path}")
            if resolved_path.is_dir():
                raise ValueError(f"Cannot delete directory. Use delete_directory() if needed.")
            
            resolved_path.unlink()

            self._log_opertion("delete", resolved_path, True, {})

            return {
                "success": True,
                "path": str(resolved_path),
            }
        except Exception as e:
            self._log_opertion("delete", path, False, {"error": str(e)})
            
            return {
                "success": False,
                "error": str(e),
                "path": str(path),
            }
            
    def get_file_info(self, path : Union[str, Path]) -> Dict[str, Any]:
        
        try:
            resolved_path = self._resolve_path(path)
            stat = resolved_path.stat() if resolved_path.exists() else None
            
            result = {
                "success": True,
                "path": str(resolved_path),
                "exists": resolved_path.exists(),
                "is_file": resolved_path.is_file() if resolved_path.exists() else False,
                "is_dir": resolved_path.is_dir() if resolved_path.exists() else False,
                "size": stat.st_size if resolved_path.exists() else 0,
            }
            if resolved_path.exists():
                result["created"] = datetime.fromtimestamp(stat.st_ctime).isoformat()
                result["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                result["accessed"] = datetime.fromtimestamp(stat.st_atime).isoformat()
            return result
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "path": str(path),
            }
            
    def get_workspace_info(self) -> Dict[str, Any]:
        
        total_files = 0
        total_size = 0
        
        for root, dirs, files in os.walk(self.workspace):
            for file in files:
                file_path = Path(root) / file
                total_files += 1
                total_size += file_path.stat().st_size
        
        return {
            "success": True,
            "workspace": str(self.workspace),
            "total_files": total_files,
            "total_size_mb": total_size / (1024 * 1024),
            "directories": {
                "inputs": str(self.inputs_dir),
                "outputs": str(self.outputs_dir),
                "temp": str(self.temp_dir),
            }
        }
    
    
def create_file_tool() -> FileSystemTool:
        
    return FileSystemTool()

        

            





        