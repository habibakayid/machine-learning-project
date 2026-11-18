---
version: alpha
name: "Flowers Classification"
description: "A specimen-card layout for a university model demo."
colors:
  primary: "#3E6851"
  background: "#F5F6ED"
  secondary: "#E7ECDE"
  text: "#202F29"
typography:
  display:
    fontFamily: "Instrument Serif, Georgia, serif"
  sans:
    fontFamily: "Streamlit sans serif"
spacing:
  page-max: "66rem"
rounded:
  image: "0.5rem"
components:
  upload: {}
  select: {}
  specimen: {}
---

# Flowers Classification design

The app is a single English-language classroom demo. A student selects a model and supplies a photo. The main result reads like a specimen card, with a prominent flower name above the photograph and a compact breakdown of the model output below.

Streamlit owns the uploader, select menu, alerts, progress, and focus states. The light theme in [config.toml](.streamlit/config.toml) owns the palette. [image_app.py](gui/image_app.py) owns the display typography, image bounds, and overall layout. The controls and result sit side by side on desktop and stack at narrow widths.

The prediction uses five labeled probability bars for supervised models. K-Means shows its assigned cluster. A loading status appears during inference and invalid images receive a readable inline error. The dataset example offers a local preview without an upload.
