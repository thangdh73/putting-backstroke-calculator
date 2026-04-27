# Putting Backstroke Calculator

Web app for predicting putt backstroke length and generating a repeatable SoundTempo-style practice tone.

## Run locally

```bash
python3 -m pip install --user -r requirements.txt
python3 -m pip install --user -r requirements-streamlit.txt
python3 -m streamlit run backstroke_app_new.py
```

The app uses SQLite by default at `backstroke_data.sqlite`. On first run, it seeds the SQL table from the existing repository data.

## Vercel deployment

This repository includes a Vercel-compatible web app:

- `public/` serves the static calculator UI from Vercel's CDN.
- `api/predict.py` runs as a Python serverless function.
- `data/backstroke_observations.sqlite` stores the backstroke observations used by the API.
- `backstroke_app_new.py` remains the Streamlit version for local development.

Deploy with:

```bash
npx vercel --prod
```

The Vercel app keeps the prediction flow and SoundTempo tone generation. No secrets are required.

## Streamlit Community Cloud deployment

Use Streamlit Community Cloud for the free hosted deployment. Vercel is optimized for static apps and serverless web functions, while this app needs Streamlit's long-running Python server and websocket connection.

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io/ and create a new app.
3. Select this repository and branch.
4. Set the main file path to `backstroke_app_new.py`.
5. Deploy.

No secrets are required for the default SQLite-backed deployment. To use a different SQLite file path, set `BACKSTROKE_SQLITE_PATH` in the host environment.
