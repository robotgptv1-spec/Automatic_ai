# AutoAI — Tabular ML Console

A single-server Flask app that trains a small PyTorch MLP on any CSV you
upload — binary classification, multi-class classification, or regression,
auto-detected from your target column.

Built on top of the `ChatGptNetwork` / `SimpleClassifier` /
`SimpleSoftmaxClassifier` code, with the training bugs fixed (see the
docstring at the top of `ml_engine.py` for the full list) and a real
upload → configure → train → predict UI wrapped around it.

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000**.

A tiny sample dataset is included at `sample_data/sample_customers.csv`
(predict `purchased` from `age`, `income`, `city`) if you want to try the
flow before uploading your own data.

## How it works

1. **Upload** — drop a `.csv`. The backend reads it with pandas and shows
   row/column counts, dtypes, and a preview.
2. **Configure** — pick a target column and which columns are features.
   Task type defaults to "auto": if the target is text or a small set of
   whole numbers it's treated as classification, otherwise regression.
3. **Train** — a small MLP (`64 → 32 → output`) trains for however many
   epochs you set. Numeric features are standardized; categorical features
   are label-encoded (unseen categories at prediction time fall back
   safely instead of crashing). The console streams per-epoch train/test
   loss and accuracy or R².
4. **Predict** — a form is generated from your feature columns
   (dropdowns for categorical columns, number inputs for numeric ones).
   Submitting it returns the predicted class + probabilities, or the
   predicted number for regression. You can also download the trained
   model + preprocessing artifacts as a `.pkl`.

## Project layout

```
app.py                 Flask routes (upload/configure/train/predict)
ml_engine.py            Model classes + AutoMLSession (preprocessing, training, inference)
templates/index.html    Single-page UI
static/css/style.css    Styling
static/js/main.js       Frontend logic — no framework, no build step
sample_data/            A tiny example CSV to try the flow with
```

## Notes / things to know

- Sessions are kept **in memory** on the server (a Python dict keyed by a
  session id). Restarting the server clears all uploaded datasets and
  trained models. That's fine for local/personal use; for multi-user
  deployment you'd want to move this to Redis or a database.
- Only `.csv` is supported right now. Excel support would just mean
  swapping `pd.read_csv` for `pd.read_excel` in `ml_engine.load_csv`.
- Max upload size is 25MB (`app.config["MAX_CONTENT_LENGTH"]`), change it
  in `app.py` if you need more.
- The MLP architecture (`64 → 32`) is fixed for now — could easily expose
  hidden-layer size as another slider if you want more control.
