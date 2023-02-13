from abc import ABC, abstractmethod

class AbstractFilter(ABC):
    """
    .. description:: 
    Abstract class defining the standard interface for a filter. A filter can be used alone or inside a
    pipeline in order to process some input. The type of input and the applied process vary from filter
    to filter. Each filter is expected to define the type of inputs it accepts and the type of output it
    returns, in order to assure valid composition of filters in a pipeline.
    """
    
    @abstractmethod
    def input_type(self):
        """
        .. description::
        Type of input the filter expects to receive when executed.
        """
        
        pass

    @abstractmethod
    def output_type(self):
        """
        .. description::
        Type of output the filter returns when executed.
        """

        pass

    @abstractmethod
    def execute(self, input, device=None):
        """
        .. description::
        Abstract method called when executing the filter on the provided input.

        .. inputs::
        input:   Input of the filter, expected to be the same type returned by method input_type.
        device:  For filters supporting execution both on cpu and gpu, the device on which 
                 to run said execution. If None, device is set to 'cuda' if execution on gpu is 
                 available or 'cpu' otherwise.
        verbose: For filters supporting this parameter, determines whether to print additional
                 info during execution or not.
        """

        pass