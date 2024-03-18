from pathlib import Path

class Logger(object):
    def __init__(self, log_file_path: str) -> None:
        
        if log_file_path is not None:
            self.log_file_path = Path(log_file_path)
            self.log_file = open(self.log_file_path, 'w')
        else:
            self.log_file = None

    def log(self, message: str) -> None:
        if self.log_file is not None:
            self.log_file.write(message + '\n')
            self.log_file.flush()
        else:
            print(message)
