import numpy as np
import shap
import transformers
from .utils import isinstance_ipython


class Model():
    """ This wraps models used in AdaTest so that have a consistent interface.

    This should eveutually just be the Model class from SHAP, but we keep a simple version here for now
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
            if isinstance_ipython(model, transformers.Pipeline):
                self.__class__ = shap.models.TransformersPipeline
                shap.models.TransformersPipeline.__init__(self, model, **kwargs)
                self.output_names = output_names
            else:
                self.inner_model = model
                self.output_names = output_names

    def __call__(self, *args):
        return np.array(self.inner_model(*args))

    
# class TransformersPipelineWrap():
#     """ This wraps the SHAP version to allow of direct output_names assignment.

#     TOTO: move direct output_names assignment support to SHAP.
#     """
#     def __init__(self, pipeline, output_names=None, rescale_to_logits=False):
#         self._inner_model = shap.models.TransformersPipeline(pipeline, rescale_to_logits)
#         self.output_names = output_names

#     @property
#     def output_names(self):
#         return self._inner_model.output_names

#     @output_names.setter
#     def output_names(self, value):
#         self._inner_model.output_names = value
#         if value is not None:
#             self._inner_model.label2id = {name: i for i, name in enumerate(value)}
#             self._inner_model.id2label = {i: name for i, name in enumerate(value)}

#     def __call__(self, strings):
#         return self._inner_model(strings)
