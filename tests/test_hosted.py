"""Request boundaries and real sample rendering; no model quota is consumed."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import fitz

import app as annotator


def text_pdf(pages=1):
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=612, height=792)
        page.insert_textbox(fitz.Rect(40, 80, 410, 650),
                            'A clear reading connects claims to evidence. A reader can ask useful questions. ' * 12,
                            fontsize=11)
    result = doc.tobytes()
    doc.close()
    return result


class HostedContractTests(unittest.TestCase):
    def setUp(self):
        self.client = annotator.app.test_client()

    def submit(self, content, **fields):
        return self.client.post('/api/process', data={
            'api_key': 'test-not-a-real-key', 'file': (io.BytesIO(content), 'reading.pdf'), **fields,
        })

    def test_limits_are_exposed_without_credentials(self):
        response = self.client.get('/api/config')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['max_pages'], 100)
        self.assertEqual(response.json['max_upload_bytes'], 100 * 1024 * 1024)
        self.assertTrue(response.json['requires_api_key'])
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertNotIn('api_key', response.json)

    def test_api_key_is_required_even_if_local_environment_has_one(self):
        with patch.object(annotator, 'DEFAULT_API_KEY', 'never-use-server-key'), patch.object(annotator.genai, 'Client') as sdk:
            response = self.client.post('/api/process', data={'file': (io.BytesIO(text_pdf()), 'reading.pdf')})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json['code'], 'api_key_required')
        sdk.assert_not_called()

    def test_invalid_file_and_model_fail_before_a_provider_request(self):
        with patch.object(annotator.genai, 'Client') as sdk:
            response = self.submit(b'not a PDF')
            model_response = self.submit(text_pdf(), model='bad\nmodel')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json['code'], 'invalid_pdf')
        self.assertEqual(model_response.json['code'], 'invalid_model')
        sdk.assert_not_called()

    def test_oversized_upload_is_rejected(self):
        response = self.submit(b'%PDF-' + b'0' * annotator.HOSTED_MAX_UPLOAD)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json['code'], 'file_too_large')

    def test_page_and_scan_limits(self):
        response = self.submit(text_pdf(101))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json['code'], 'page_limit')
        doc = fitz.open()
        doc.new_page()
        scan_response = self.submit(doc.tobytes())
        doc.close()
        self.assertEqual(scan_response.json['code'], 'text_pdf_required')

    def test_encrypted_pdf_is_rejected_without_a_password_prompt(self):
        doc = fitz.open(stream=text_pdf(), filetype='pdf')
        data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw='owner', user_pw='reader')
        doc.close()
        response = self.submit(data)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json['code'], 'encrypted_pdf')

    def test_upstream_failure_does_not_return_credentials_or_retain_jobs(self):
        prior_jobs = set(annotator.jobs)
        secret = 'credential-not-for-response'
        with patch.object(annotator, 'generate_annotations', new=AsyncMock(side_effect=RuntimeError('Gemini API error ' + secret))):
            response = self.submit(text_pdf())
        self.assertEqual(response.status_code, 502)
        self.assertNotIn(secret.encode(), response.data)
        self.assertEqual(response.json['code'], 'provider_failed')
        self.assertEqual(set(annotator.jobs), prior_jobs)

    def test_sample_renders_a_real_pdf_without_ai_and_cleans_temp_files(self):
        engine_folders = []
        original = annotator.ensure_handwriter

        def observe_engine():
            node, bridge = original()
            engine_folders.append(bridge.parent.parent)
            return node, bridge

        prior_jobs = set(annotator.jobs)
        with patch.object(annotator.genai, 'Client', side_effect=AssertionError('Demo must never call AI')), patch.object(annotator, 'ensure_handwriter', side_effect=observe_engine):
            response = self.client.post('/api/demo')
        self.assertEqual(response.status_code, 200, response.data[:200])
        self.assertEqual(response.mimetype, 'application/pdf')
        self.assertEqual(response.headers['X-AutoAnnotator-Demo'], 'true')
        self.assertIn('no-store', response.headers['Cache-Control'])
        self.assertLess(len(response.data), annotator.HOSTED_MAX_OUTPUT)
        self.assertGreater(len(response.data), 5000)
        with fitz.open(stream=response.data, filetype='pdf') as doc:
            self.assertEqual(len(doc), 1)
            self.assertEqual(tuple(doc[0].rect), (0.0, 0.0, 612.0, 792.0))
            # The original renderer flattens the page for printing, including the
            # visible DEMO label and actual highlights/handwriting.
            self.assertGreater(len(doc[0].get_images()), 0)
        self.assertEqual(set(annotator.jobs), prior_jobs)
        self.assertTrue(engine_folders)
        self.assertTrue(all(not directory.exists() for directory in engine_folders))

    def test_preview_renders_image_only_pdf_without_ai_or_retained_jobs(self):
        # Finished downloads are flattened scans, so the preview must not apply
        # the selectable-prose upload gate used by the AI endpoint.
        from PIL import Image
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        image_bytes = io.BytesIO()
        Image.new("RGB", (80, 80), "magenta").save(image_bytes, format="PNG")
        page.insert_image(fitz.Rect(30, 30, 210, 210), stream=image_bytes.getvalue())
        data = doc.tobytes()
        doc.close()
        prior_jobs = set(annotator.jobs)
        with patch.object(annotator.genai, 'Client', side_effect=AssertionError('Preview must never call AI')):
            response = self.client.post('/api/preview', data={'file': (io.BytesIO(data), '/private/reading.pdf')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'image/png')
        with Image.open(io.BytesIO(response.data)) as image:
            self.assertEqual(image.size, (765, 990))
            self.assertEqual(image.getpixel((100, 100)), (255, 0, 255))
        self.assertIn('no-store', response.headers['Cache-Control'])
        self.assertNotIn('/private', response.headers.get('Content-Disposition', ''))
        self.assertEqual(set(annotator.jobs), prior_jobs)

    def test_preview_rejects_invalid_or_unbounded_page_geometry(self):
        bad = self.client.post('/api/preview', data={'file': (io.BytesIO(b'no PDF'), 'reading.pdf')})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(bad.json['code'], 'invalid_pdf')
        doc = fitz.open()
        doc.new_page(width=20000, height=20000)
        huge_page = self.client.post('/api/preview', data={'file': (io.BytesIO(doc.tobytes()), 'huge.pdf')})
        doc.close()
        self.assertEqual(huge_page.status_code, 400)
        self.assertEqual(huge_page.json['code'], 'page_size')

    def test_hosted_mode_does_not_offer_a_ram_background_queue(self):
        with patch.object(annotator, 'HOSTED', True):
            response = self.client.post('/api/annotate')
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json['code'], 'use_process_endpoint')

    def test_annotate_page_retries_on_freeze_timeout_and_succeeds(self):
        import asyncio
        from unittest.mock import MagicMock
        annotator.jobs['test_job_id'] = {}
        fake_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"pages": [{"page_number": 1, "highlights": [], "margin_notes": []}]}'

        # First call times out (freeze simulation), second call succeeds
        call_count = [0]
        async def fake_generate_content(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise asyncio.TimeoutError("Simulated API freeze timeout")
            return mock_response

        fake_client.models.generate_content = fake_generate_content

        result = asyncio.run(annotator.annotate_page(
            fake_client, 1, "Sample text for page", "test_job_id", [0.0]
        ))
        self.assertEqual(result.page_number, 1)
        self.assertEqual(call_count[0], 2)

    def test_annotate_page_graceful_fallback_if_all_retries_exhausted(self):
        import asyncio
        from unittest.mock import MagicMock
        annotator.jobs['test_job_id'] = {}
        fake_client = MagicMock()

        # All calls time out or error
        async def fake_generate_content(*args, **kwargs):
            raise asyncio.TimeoutError("Persistent freeze")

        fake_client.models.generate_content = fake_generate_content

        with patch.object(annotator, 'MAX_ATTEMPTS', 2):
            result = asyncio.run(annotator.annotate_page(
                fake_client, 5, "Sample text for stubborn page", "test_job_id", [0.0]
            ))
        # Should gracefully return empty page annotations so whole 100-page book is not dropped
        self.assertEqual(result.page_number, 5)
        self.assertEqual(result.highlights, [])
        self.assertEqual(result.margin_notes, [])


if __name__ == '__main__':
    unittest.main()
