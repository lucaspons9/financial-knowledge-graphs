"""
Data models for OpenAI batch processing.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BatchResult:
    """Information about a single batch processing result."""
    item_id: str
    result_file: str
    parsed_content: Dict[str, Any]

@dataclass
class BatchInfo:
    """Information about a batch job."""
    batch_id: str
    created_at: datetime
    status: str
    n_items: int
    expires_at: Optional[datetime] = None
    original_texts: Optional[Dict[str, str]] = None
    task: Optional[str] = None
    model: Optional[str] = None
    file_id: Optional[str] = None
    using_custom_ids: bool = False
    completed_at: Optional[datetime] = None
    results_path: Optional[str] = None
    n_results: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchInfo':
        """Create a BatchInfo instance from a dictionary."""
        # Convert string timestamps to datetime objects with proper type checking
        created_at_str = data.get('created_at')
        created_at = datetime.fromisoformat(created_at_str) if isinstance(created_at_str, str) else datetime.now()
        
        expires_at_str = data.get('expires_at')
        expires_at = datetime.fromisoformat(expires_at_str) if isinstance(expires_at_str, str) else None
        
        completed_at_str = data.get('completed_at')
        completed_at = datetime.fromisoformat(completed_at_str) if isinstance(completed_at_str, str) else None
        
        return cls(
            batch_id=data.get('batch_id', ''),
            created_at=created_at,
            status=data.get('status', 'unknown'),
            n_items=data.get('n_items', 0),
            expires_at=expires_at,
            original_texts=data.get('original_texts'),
            task=data.get('task'),
            model=data.get('model'),
            file_id=data.get('file_id'),
            using_custom_ids=data.get('using_custom_ids', False),
            completed_at=completed_at,
            results_path=data.get('results_path'),
            n_results=data.get('n_results')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the BatchInfo instance to a dictionary."""
        return {
            'batch_id': self.batch_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'status': self.status,
            'n_items': self.n_items,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'original_texts': self.original_texts,
            'task': self.task,
            'model': self.model,
            'file_id': self.file_id,
            'using_custom_ids': self.using_custom_ids,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'results_path': self.results_path,
            'n_results': self.n_results
        }

@dataclass
class BatchStatusInfo:
    """Status information for a batch job."""
    batch_id: str
    status: str
    completed: bool
    created_at: str
    last_checked: str
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    error: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchStatusInfo':
        """Create a BatchStatusInfo instance from a dictionary."""
        return cls(
            batch_id=data.get('batch_id', ''),
            status=data.get('status', 'unknown'),
            completed=data.get('completed', False),
            created_at=data.get('created_at', ''),
            last_checked=data.get('last_checked', ''),
            output_file_id=data.get('output_file_id'),
            error_file_id=data.get('error_file_id'),
            error=data.get('error')
        )

# ParentBatchInfo class removed - no longer needed 