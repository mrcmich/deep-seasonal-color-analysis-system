from .abstract_pipeline import AbstractPipeline

class Pipeline(AbstractPipeline):
    def __init__(self):
        self.filters = []

    def filters(self):
        """
        .. description:: 
        Method returning the list of filters composing the pipeline.
        """

        return self.filters

    def add_filter(self, filter):
        assert(len(self.filters) == 0 or self.filters[-1].output_type() == filter.input_type())
        self.filters.append(filter)

    def execute(self, input):
        last_output = input

        for filter in self.filters:
            current_output = filter.execute(last_output)
            last_output = current_output
        
        return last_output