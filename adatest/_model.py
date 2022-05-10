import numpy as np
import shap
import transformers
from .utils import isinstance_ipython


class Model():
    """ This wraps models used in AdaTest so that have a consistent interface.

    This should eventually just be the Model class from SHAP, but we keep a simple version here for now
    so we can easily update it during initial development.
    """

    def __new__(cls, model, *args, **kwargs):
        """ If we are wrapping a model that is already a Model, we just return it.
        """
        if isinstance_ipython(model, Model) or isinstance_ipython(model, shap.models.Model):
            return model
        else:
            return super().__new__(cls)
    
    def __init__(self, model, output_names=None, **kwargs):
        """ Build a new model by wrapping the given model object.

        Parameters
        ----------
        model : object
            The model to wrap. This can be a plain python function that accepts a list of strings and returns either
            a vector of probabilities or another string. It can also be a transformers pipeline object (we try to wrap
            common model types transparently).

        output_names : list of str, optional
            The names of the outputs of the model. If not given, we try to infer them from the model.
        """

        # finish early if we are wrapping an object that is already a Model
        if isinstance_ipython(model, Model) or isinstance_ipython(model, shap.models.Model):
            if output_names is not None:
                self.output_names = output_names
            assert len(kwargs) == 0
            return

        # get outputs names from the model if it has them and we don't
        if output_names is None and hasattr(model, "output_names"):
            output_names = model.output_names

        # If we are in the base class we check to see if we should rebuild the model as a specialized subclass
        if self.__class__ is Model:
            
            # wrap transformer pipeline objects for convenience
            if isinstance_ipython(model, transformers.pipelines.text_classification.TextClassificationPipeline):
                self.__class__ = shap.models.TransformersPipeline
                shap.models.TransformersPipeline.__init__(self, model, **kwargs)
                if output_names is not None: # Override output names if user supplied
                    self.output_names = output_names

            elif isinstance_ipython(model, transformers.pipelines.text_generation.TextGenerationPipeline):
                self.__class__ = TransformersTextGenerationPipeline
                TransformersTextGenerationPipeline.__init__(self, model, **kwargs)
            
            else:
                self.inner_model = model
                self.output_names = output_names

    def __call__(self, *args, **kwargs):
        return np.array(self.inner_model(*args, **kwargs))

    
class TransformersTextGenerationPipeline():
    """ This wraps the transformer text generation pipeline object to match the Model API.

    TOTO: move this to SHAP.
    """
    def __init__(self, pipeline):
        self._inner_model = pipeline
        self.output_names = None

    def __call__(self, strings, completions=1):
        full_out = []
        for c in range(completions):
            inner_out = self._inner_model(strings)
            out = []
            for s, data in zip(strings, inner_out):
                out.append(data[0]["generated_text"][len(s):]) # remove the input text from the output
            full_out.append(out)
        return np.array(full_out).T
