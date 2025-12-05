GitHub: https://github.com/thedraken/a4s-eval-thomas-mortimer
GitLab: https://gitlab.com/uniluxembourg/Personalfolders/thomas.mortimer.001/a-4-s-eval-thomas-mortimer

Extra libraries installed via UV Sync, compared to main project:
"textattack",
"tensorflow-datasets",
"stanza",
"sentence_transformers==5.1.2"

python generate_imdb_csv.py --limit 1000 --output imdb_transformed.csv
python run_eval.py --csv imdb_transformed.csv --model hf