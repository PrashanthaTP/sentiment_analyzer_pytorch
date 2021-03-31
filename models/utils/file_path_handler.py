"""
classes
--------------
+ FilePathHandler : container for metrics_file and model_file
"""
class FilePathHandler:
    """container for metrics_file and model_file
    """
    def __init__(self,metrics_file,model_file):
        self.metrics_file = metrics_file
        self.model_file = model_file
        
        
        