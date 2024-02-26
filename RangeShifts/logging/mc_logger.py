import json
import time

class MC_logger:
    def __init__(self, log_filename='mc_log.json'):
        self.log_filename = log_filename
        self.log_data = []
        
    def log_function_call(self, function_name, args, range_collection_name):
        log_entry = {
            "timestamp": time.time(),
            "log_level": "INFO",
            "message": {'function' : function_name,
                        'args': args,
                        'RangeCollection_name': range_collection_name
                       }
        }
        self.log_data.append(log_entry)
        self._save_log_data()

    def log_output(self, function_name, output, estimation_values=None):
        log_entry = {
            "timestamp": time.time(),
            "log_level": "INFO",
            "message": {'output': output,
                        'estimation_values': estimation_values
                       }
        }
        
        self.log_data.append(log_entry)
        self._save_log_data()



    def _save_log_data(self):
        with open(self.log_filename, 'w') as file:
            json.dump(self.log_data, file, indent=4)

def initialize_logging():
    mc_logger = MC_logger()
    # Optionally log initial setup information
    mc_logger.log_output("Initialization", "Logging initialized.")
    return mc_logger

# Call the initialization function when the module is imported
mc_logger = initialize_logging()
