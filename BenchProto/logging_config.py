import logging


TEST_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    
    # --- DEFINE FORMATS ---
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'simple': {
            'format': '[%(levelname)s] %(message)s'
        },
    },
    
    # --- DEFINE HANDLERS (Destinations) ---
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout',
        },
        'test_file': {
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'filename': 'BenchProtoLogFile.log',
            'mode': 'w', 
        },
    },
    
    # --- CONFIGURE LOGGERS ---
    'loggers': {
        
        # 1. Your code's logger
        'BenchmarkingFactory': {
            'level': 'DEBUG',
            'handlers': ['console', 'test_file'],
            'propagate': False 
        },

        'main': {  
        'level': 'DEBUG',
        'handlers': ['console', 'test_file'],
        'propagate': False
        },

        'BenchmarkingFactory.aiModel': {
        'level': 'DEBUG',
        'handlers': ['console', 'test_file'],
        'propagate': False
        },

        # 2. Silencing onnxruntime
        'onnxruntime': {
            'level': 'ERROR', # Only show ERRORs or higher
            'handlers': ['console', 'test_file'],
            'propagate': False # Stop logs from going to root
        },
        
        # 3. Silencing onnxscript
        'onnxscript': {
            'level': 'ERROR', # Only show ERRORs or higher
            'handlers': ['console', 'test_file'],
            'propagate': False # Stop logs from going to root
        }
    },
    
    # --- THE DEFAULT (ROOT) LOGGER ---
    'root': {
        'level': 'DEBUG', # Default for all other libraries
        'handlers': ['console', 'test_file']
    }
}