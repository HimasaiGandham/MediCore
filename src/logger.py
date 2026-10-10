import logging
import os
import json
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path

# Setup logs directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "medicore.log"

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        if hasattr(record, "audit_metadata"):
            log_record.update(record.audit_metadata)
        
        # Safely handle exception if present without logging raw data
        if record.exc_info:
            log_record["exception_type"] = record.exc_info[0].__name__
            log_record["safe_error"] = str(record.exc_info[1]).split('\n')[0][:200]
            
        return json.dumps(log_record)

def setup_logger(name="medicore_audit"):
    """
    Sets up a structured JSON logger for operational audit trails.
    """
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    formatter = JsonFormatter()

    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Note: StreamHandler is optional but good for local dev
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

audit_logger = setup_logger()

def generate_request_id() -> str:
    """Generates a unique request ID for correlation across query stages."""
    return str(uuid.uuid4())

def log_system_event(event_name: str, status: str, details: dict = None, level: str = "info"):
    """
    Logs general system lifecycle events (startup, index load/rebuild, API errors).
    """
    metadata = {
        "event_name": event_name,
        "status": status,
    }
    if details:
        metadata.update(details)
        
    extra = {"audit_metadata": metadata}
    msg = f"{event_name} - {status}"
    
    if level.lower() == "error":
        audit_logger.error(msg, extra=extra)
    elif level.lower() == "warning":
        audit_logger.warning(msg, extra=extra)
    else:
        audit_logger.info(msg, extra=extra)

def log_query_lifecycle(
    request_id: str,
    event_name: str,
    status: str,
    duration_ms: float = 0.0,
    num_results: int = 0,
    engine: str = "auto",
    error_category: str = None
):
    """
    Logs operational metadata for a query lifecycle without recording the query text itself.
    """
    metadata = {
        "request_id": request_id,
        "event_name": event_name,
        "status": status,
        "engine": engine,
        "retrieval_count": num_results,
    }
    if duration_ms > 0:
        metadata["duration_ms"] = round(duration_ms, 2)
    if error_category:
        metadata["error_category"] = error_category
        
    extra = {"audit_metadata": metadata}
    msg = f"Request {request_id[:8]} - {event_name} - {status}"
    
    if error_category:
        audit_logger.error(msg, extra=extra)
    else:
        audit_logger.info(msg, extra=extra)
