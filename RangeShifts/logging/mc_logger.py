import json
import time

class MC_logger:
    def __init__(self, log_filename='mc_log.json'):
        self.log_filename = log_filename
        self.log_data = []
        
    def log_function_call(self, function_name, args, range_collection_name,**log_kwargs):
        log_entry = {
            "timestamp": time.time(),
            "log_level": "INFO",
            "message": {'function' : function_name,
                        'args': args,
                        **log_kwargs
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


class LoggerSingleton:
    '''
    Singleton class to manage the instance of the logger used throughout the simulations.
    Reduntant for now. Might be usefull for the future, if more flexibility at the instantaniation is needed.
    '''
    _instance = None

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = MC_logger()
        return cls._instance

def initialize_logging():
    mc_logger = LoggerSingleton.get_instance()
    # Optionally log initial setup information
    mc_logger.log_output("Initialization", "Logging initialized.")
    return mc_logger

# Call the initialization function when the module is imported
mc_logger = initialize_logging()
