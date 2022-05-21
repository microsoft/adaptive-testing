# AdaTest (alpha preview release)
AdaTest uses language models against themselves to build suites of unit tests. It is an interative (and fun!) process between a user and a language model that results in a tree of unit tests specifically adapted to the model you are testing. Fixing any failed tests with fine-tuning then leads to an iterative debugging process similar to traditional software development.

<p align="center">
  <img src="docs/images/main_loops.png" width="300" alt="AdaTest loops" />
</p>

## Install

```
pip install adatest
```

## Sentiment analysis example

How to test a simple two-way sentiment analysis model using AdaTest running in a Jupyter notebook (full notebook [here](here)).

```python
import transformers
import adatest

# create a HuggingFace sentiment analysis model
classifier = transformers.pipeline("sentiment-analysis", return_all_scores=True)

# set AdaTest's language model backend (HuggingFace and AI21 also supported)
generator = adatest.generators.OpenAI('davinci', api_key=OPENAI_API_KEY)

# optional: load tests from a dataset
dataset_tree = adatest.TestTree((X, y), compute_embeddings=True)

# load a starting tree of tests targeted at sentiment analysis
tests = adatest.TestTree("test_trees/sentiment_analysis/two_way_demo.csv")

# apply the tests to our model to launch a notebook-based testing interface
tests(classifier, generators={'curie': generator, 'dataset': dataset_tree}, auto_save=True) # wrap with adatest.serve to launch a standalone server
```

**Image showing the root and scores for the basic two way sentiment tree.**

Once we have opened a test tree browser, we can navigate to specific topics of interest and then add new tests specifically targeted at our current model. Clicking 'Suggest tests` in a topic proposes a large number of new tests targeted at the current model that we can add to the test tree if we like.

**Image showing many suggestions in a topic where the current model does not fail.**

After multiple rounds of test suggestions AdaTest learns to find failures of the target model within the current topic.

**Image showing lots of now-failing tests.**

Once we have created enough new tests we can organize them into new topics or switch to another topic and continue the testing process. After we have found enough failed tests in the *testing loop*, we can then fine tune the model on the tests to fix the errors we have found. To prevent catastrophic forgetting we fine tune on a 50/50 mix of tests and samples from the original fine-tuning dataset of the model.

```python
# create a new sentiment model that fixes the problems we found
tensor_output_model2 = fine_tune(tensor_output_model, tests) # see sample notebook for definition

# apply the tests to the new model
tests(tensor_output_model2)
```

**Image showing the root and scores for the new model in the basic two way sentiment tree.**

Note that almost all tests now pass in the new model, but that does not mean the model is perfect! Since we have used our tests as training data we need to create new tests to properly evaluate our model and make sure we have not missed problems or created new problems.

**Image showing the root and scores for the new model in the basic two way sentiment tree with more errors after suggestions.**

After letting AdaTest suggest new tests we find that there are still remaining errors in the model. So we can repeat the fine tuning process to fix these new issues. This will again result in a passing set of tests. The debugging process can be repeated as long as desired, iteratively fixing bugs and so improving the target model performance for the capabilities measured in the test tree.


## Translation example

AdaTest can test any machine learning model that takes text as input, even models only accessable through an API. Here we demonstrate how to test the Azure Translation API using AdaTest.






## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
