import sys
import logging
import subprocess
from datetime import datetime, timezone

class RedirectPrintToLogging:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, text):
        for line in text.rstrip().split('\n'):
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

def execute_shell_command(command_string : str, verbose=1) -> bool:
    """
    Execute shell commands
    
    Parameters
    ----------
    command_string : str
        Command to execute
    verbose: int, default=1
        log verbose statements
        
    Returns
    -------
    bool: True/False
    
    """
    try:
        _mv_process = subprocess.Popen(command_string,
                                        shell=True,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
        stdout, stderr = _mv_process.communicate()
    except (OSError, KeyboardInterrupt) as e:
        _mv_process.kill()

    if _mv_process.returncode == 0:
        # if there is no command output then return True
        command_output = stdout.decode('utf-8')
        if command_output:
            return command_output
        else:
            return True

    if verbose >= 1:
        logging.error(stderr.decode('utf-8'))
        time_record = datetime.now(timezone.utc)
        logging.error(f"[{time_record}] Moving study failed")

    return False

def initialize_logging(name: str) -> str:
    """
    Initialize logging
    
    Parameters
    ----------
    name : str
        Name of the logger
        
    Returns
    -------
    log_file_path: str
    
    """
    # create file path by name and current timestamp in utc
    log_file_path = f"./{name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[
                            logging.FileHandler(log_file_path),  # Log to a file
                            logging.StreamHandler()  # Log to the console
                        ])
    logger = logging.getLogger()
    sys.stdout = RedirectPrintToLogging(logger, logging.INFO)
    sys.stderr = RedirectPrintToLogging(logger, logging.ERROR)
    
    return log_file_path
    
def save_log_to_gcs(log_file_path: str, gcs_path: str) -> bool:
    """
    Save log file to GCS
    
    Parameters
    ----------
    log_file_path : str
        Path to log file
    gcs_path : str
        GCS path to save log file
        
    Returns
    -------
    bool: True/False
    
    """
    command = f"gsutil -m cp -r {log_file_path} {gcs_path}/"
    status = execute_shell_command(command)
    
    # remove log file
    command = f"rm {log_file_path}"
    status = execute_shell_command(command)
    
    return status

