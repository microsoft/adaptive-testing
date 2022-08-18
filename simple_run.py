import adatest
import os

# Used for image request routing
static_dir = os.path.join(os.curdir, "static")
image_dir = "images"
static_image_dir = os.path.join(static_dir, image_dir)
print("static image dir", static_image_dir)
os.makedirs(static_image_dir, exist_ok=True)

# Make scorer and text generator
def fake_model(prompts: list):
  return prompts
scorer = adatest.ImageScorer(fake_model, static_image_dir)
openai_generator = adatest.generators.OpenAI('curie', api_key=os.getenv('OPENAI_API_KEY'))

tests = adatest.TestTree(r"test_trees/dalleproto.csv")

print("Starting server")
adatest.serve(tests.adapt(scorer, {"openai": openai_generator}, auto_save=True, max_suggestions=3, output_mode="image", image_dir=image_dir), static_dir=static_dir, port=5000)
