from .abstract_pipeline import AbstractPipeline

class Pipeline(AbstractPipeline):
    """
    .. description:: 
    Implementation of abstract class AbstractPipeline. A pipeline is a sequence
    of composable filter executed one after the other, such that the input of each filter coincides
    with the output of the previous one.
    """
    
    def __init__(self):
        """
        .. description:: 
        Class constructor.
        """
        
        self.filters = []

    def filters(self):
        """
        .. description:: 
        Method returning the list of filters composing the pipeline.
        """

        return self.filters

    def add_filter(self, filter):
        """
        .. description::
        Method to add a new filter to the pipeline. The new filter can't be added unless its input type is
        the same as the output type of the last filter in the pipeline.

        .. inputs::
        filter: A filter. Filters are expected to implement AbstractFilter's interface.
        """
        
        assert(len(self.filters) == 0 or self.filters[-1].output_type() == filter.output_type())
        self.filters.append(filter)

    def execute(self, input):
        """
        .. description::
        Method to execute all of the pipeline's filters one after the other, in the order
        in which they were added.

        .. inputs::
        input: Input of the pipeline, coinciding with the input of the first filter.
        """

        last_output = input

        for filter in self.filters:
            current_output = filter.execute(last_output)
            last_output = current_output
        
        return last_output