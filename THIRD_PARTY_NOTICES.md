# Third-party notices

Auto Annotator was assembled and developed by John Saleh. It uses the following third-party projects and services. Dependency licenses apply to their respective components; this notice does not select a blanket license for John's original application code.

## Handwriter neural handwriting engine and weights

- Project: [Eeman1113/handwriter](https://github.com/Eeman1113/handwriter)
- Pinned revision: [`3c11fe12014d510bd5e74a91aca96aa5f772ca57`](https://github.com/Eeman1113/handwriter/tree/3c11fe12014d510bd5e74a91aca96aa5f772ca57)
- Included components: `src/lib/handwriting.js` (stored locally as `handwriting.mjs`) and `public/d.bin` pretrained weights.
- Upstream copyright: Copyright (c) 2026 Eeman Majumder.
- Upstream license: MIT. The exact upstream text is preserved in [vendor/handwriter/LICENSE](vendor/handwriter/LICENSE).

The upstream README credits Sean Vasquez's [calligrapher.ai](https://www.calligrapher.ai) implementation and Alex Graves' [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) (2013), section 5. Handwriter adapts that recurrent mixture-density network into a dependency-free module. Auto Annotator uses the pretrained model to generate pen-stroke geometry; John did not train this model.

The repository's pinned upstream MIT notice is retained for the vendored engine and weights. No claim is made that the model's training data was created by John, or that its output is a person's actual handwriting.

## Runtime dependencies

| Component | Purpose | License / source |
|---|---|---|
| Flask and Werkzeug | Web application and request handling | [BSD-3-Clause](https://github.com/pallets/flask/blob/main/LICENSE.txt) and corresponding Werkzeug license |
| PyMuPDF / MuPDF | PDF extraction, geometry, OCR integration, and rendering | [AGPL-3.0 or commercial license](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright) |
| Pillow | Image processing and pen-stroke rasterization | [HPND](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| Pydantic | Structured annotation validation | [MIT](https://github.com/pydantic/pydantic/blob/main/LICENSE) |
| Google Gen AI Python SDK | Gemini API client | [Apache-2.0](https://github.com/googleapis/python-genai/blob/main/LICENSE) |
| Node.js / nodejs-wheel-binaries | JavaScript handwriting runtime | [Node.js MIT and bundled third-party notices](https://github.com/nodejs/node/blob/main/LICENSE); [wheel distribution](https://pypi.org/project/nodejs-wheel-binaries/) |
| Tesseract (optional, local install) | OCR for scanned PDFs | [Apache-2.0](https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE) |

Transitive dependencies retain their own notices and licenses. Tesseract and its language data are not bundled with the hosted app.

## External AI service

Gemini is provided by Google. AI annotation calls require a user-supplied key or a locally configured key and remain subject to the provider's availability, quotas, billing, and terms. The fixed sample demo does not call Gemini. Auto Annotator is an independent project, not an official Google, Handwriter, or Princeton University product.
