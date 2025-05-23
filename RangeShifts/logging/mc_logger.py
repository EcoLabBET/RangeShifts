import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class MC_logger:
    def __init__(self, log_filename='mc_log.json'):
        self.log_filename = log_filename
        self.log_data = []
        
    def log_function_call(self, function_name, args,log_kwargs={}):
        log_entry = {
            "timestamp": time.time(),
            "log_level": "INFO",
            "message": {'function' : function_name,
                        'args': args,
                        'log_kwargs': log_kwargs
                       }
        }
        self.log_data.append(log_entry)
        self._save_log_data()

    def log_output(self, function_name, output, estimation_values=None,log_kwargs={}):
        log_entry = {
            "timestamp": time.time(),
            "log_level": "INFO-OUTPUT",
            "message": {'output': output,
                        'estimation_values': estimation_values,
                        'log_kwargs': log_kwargs
                       }
        }
        
        self.log_data.append(log_entry)
        self._save_log_data()

    def load_logs(self, filename=None):
        """Load logs from a JSON file."""
        filename = filename or self.log_filename
        try:
            with open(filename, 'r') as f:
                self.log_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Log file {filename} not found")

 ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
 ## USE THE FOLLOWING WITH CAUTION, AS IN THE FUTURE batch saving
 ## might be introduced to prevent memmory explosion. get_output_logs()
 ## will need to draw data from the .json file

    def get_output_logs(self):
            return [entry for entry in self.log_data if entry['log_level'] == "INFO-OUTPUT"]

    def plot_output_logs(self, pdf_filename='output_logs.pdf'):
        output_logs = self.get_output_logs()
        num_rows, num_cols = 5, 3
        entries_per_page = num_rows * num_cols
    
        with PdfPages(pdf_filename) as pdf:
            num_pages_needed = np.ceil(len(output_logs) / entries_per_page)
            for page_num, log_entry in enumerate(output_logs):
                if page_num % entries_per_page == 0:
                    fig, axs = plt.subplots(num_rows, num_cols, figsize=(9, 12))  # Define grid size based on num_rows and num_cols
                    fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust spacing
    
                row = (page_num // num_cols) % num_rows
                col = page_num % num_cols
    
                output = log_entry['message']['output']
                estimation_values = log_entry['message']['estimation_values']
                name = log_entry['message']['log_kwargs'].get('name', 'No Name')
    
                ax = axs[row, col]
                ax.plot(np.arange(100, len(estimation_values) * 100 + 1, 100), estimation_values)
                ax.axhline(y=output['p_value'], color='black', linestyle='--')
                ax.set_xlabel('Iterations')
                ax.set_ylabel('Estimation Value')
                ax.set_title(f'{name}')
                ax.grid(True)
    
                if (page_num + 1) % entries_per_page == 0 or page_num == len(output_logs) - 1:
                    pdf.savefig(fig)
                    plt.close(fig)
    
            if len(output_logs) % entries_per_page != 0:
                pdf.savefig(fig)
                plt.close(fig)


    def reset(self):
        self.log_data = []
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
    return mc_logger

# Call the initialization function when the module is imported
mc_logger = initialize_logging()
