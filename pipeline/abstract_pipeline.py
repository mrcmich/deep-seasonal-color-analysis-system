from abc import ABC, abstractmethod

class AbstractPipeline(ABC):
    """
    .. description:: 
    Abstract class defining the standard interface for a pipeline. A pipeline is a sequence
    of composable filters executed one after the other, such that the input of each filter coincides
    with the output of the previous one.
    """
    
    @abstractmethod
    def add_filter(self, filter):
        """
        .. description::
        Abstract method to add a new filter to the pipeline.

        .. inputs::
        filter: A filter. Filters are expected to implement AbstractFilter's interface.
        """
        
        pass

    @abstractmethod
    def execute(self, input, device=None, verbose=False):
        """
        .. description::
        Abstract method to execute all of the pipeline's filters one after the other, in the order
        in which they were added.

        .. inputs::
        input:   Input of the pipeline, coinciding with the input of the first filter.
        device:  For pipeline filters supporting execution both on cpu and gpu, the device on which 
                 to run said execution. If None, device is set to 'cuda' if execution on gpu is 
                 available or 'cpu' otherwise.
        verbose: For pipeline filters supporting this parameter, determines whether to print additional
                 info during execution or not.
        """

        pass