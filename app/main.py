from typing import Union

from fastapi import FastAPI
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

app = FastAPI()

tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

@app.get("/")
def read_root():

    text = """
    The problem of vertical flow stability in an oil reservoir with a gas cap is considered, when the oil flow obeys the Brinkman equation. Boundary conditions at the moving boundary of the gas-oil interface are derived and a basic solution is obtained. The normal mode method is used to study the stability of the gas–oil interface. The obtained dispersion equation is investigated. Conditions for flow stability are found for all values of the parameters, and it is shown that, in the linear approximation, the growth rate of short-wave perturbations tends to zero with increasing wave number.
    """

    print(text)

    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")

    summary = model.generate(**tokens)

    return { "Hello": summary[0] }

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
