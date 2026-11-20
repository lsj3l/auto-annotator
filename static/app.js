'use strict';

(() => {
  const byId = id => document.getElementById(id);
  const form = byId('annotation-form');
  const fileInput = byId('pdf-file');
  const keyInput = byId('api-key');
  const dropZone = byId('drop-zone');
  const fullLocal = byId('full-local');
  const regenerate = byId('regenerate');
  const startButton = byId('annotate-button');
  const demoButton = byId('demo-button');
  const emptyState = byId('empty-state');
  const processingState = byId('processing-state');
  const progress = byId('progress');
  const statusBox = byId('status-box');
  const statusText = byId('status-text');
  const result = byId('pdf-result');
  const output = byId('output-card');
  const badge = byId('result-type');
  let config = null;
  let busy = false;
  let resultUrl = null;
  let previewUrl = null;

  const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
  const localMode = () => config && !config.hosted && fullLocal.checked;
  const formatSize = bytes => bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KiB` : `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;

  function showStatus(message, isError = false) {
    statusText.textContent = message;
    statusBox.hidden = false;
    statusBox.classList.toggle('error', isError);
    statusBox.setAttribute('role', isError ? 'alert' : 'status');
  }

  function showFile() {
    const file = fileInput.files[0];
    const summary = byId('file-summary');
    summary.hidden = !file;
    summary.textContent = file ? `${file.name} · ${formatSize(file.size)}` : '';
  }

  function updateMode() {
    const isLocal = localMode();
    byId('limit-text').textContent = isLocal ? 'Local queue · larger PDFs · optional OCR' : `Text-based readings · up to ${config?.max_pages || 100} pages · ${formatSize(config?.max_upload_bytes || 104857600)}`;
    byId('file-guidance').textContent = isLocal ? 'Scans require a local Tesseract installation.' : 'A selectable-text PDF works best.';
    byId('regenerate-option').hidden = !isLocal;
    byId('key-guidance').textContent = isLocal ? 'Leave blank only if GEMINI_API_KEY is configured locally. AI requests use your Google account’s quota and billing.' : 'Not saved in your browser. AI requests use your Google account’s quota and billing.';
  }

  function setBusy(value, isDemo = false) {
    busy = value;
    for (const control of [startButton, demoButton, fileInput, keyInput, fullLocal, regenerate]) control.disabled = value || !config;
    form.setAttribute('aria-busy', String(value));
    output.setAttribute('aria-busy', String(value));
    processingState.hidden = !value;
    if (value) {
      if (resultUrl) URL.revokeObjectURL(resultUrl);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      resultUrl = null;
      previewUrl = null;
      result.hidden = true;
      byId('pdf-preview').removeAttribute('src');
      byId('pdf-preview').hidden = true;
      byId('download-link').removeAttribute('href');
      byId('open-link').removeAttribute('href');
      emptyState.hidden = true;
      statusBox.hidden = true;
      badge.hidden = true;
      progress.removeAttribute('value');
      byId('processing-title').textContent = isDemo ? 'Writing a sample in the margins…' : 'Making room for thought…';
      byId('output-subtitle').textContent = isDemo ? 'Rendering preset sample notes. No AI request.' : 'Processing your reading. Keep this page open.';
    }
  }

  async function request(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1800000);
    try {
      const response = await fetch(url, {...options, signal: controller.signal, cache: 'no-store'});
      if (!response.ok) {
        let message;
        if ((response.headers.get('content-type') || '').includes('application/json')) {
          const data = await response.json().catch(() => null);
          message = data?.error;
        }
        if (!message) {
          message = response.status === 413 ? 'The file or completed PDF is too large for this route. Try a smaller reading.' : response.status === 504 ? 'Processing exceeded the server time limit. Try a shorter reading or the local workflow.' : `The request could not be completed (HTTP ${response.status}). Please try again.`;
        }
        throw new Error(message);
      }
      return response;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('Processing took too long. Try a shorter reading or the local workflow.');
      if (error instanceof TypeError) throw new Error('The app could not reach the server. Check your connection and try again.');
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  async function displayPdf(response, filename, isDemo) {
    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('application/pdf')) throw new Error('The server did not return a PDF. Please try again.');
    const blob = await response.blob();
    if (!blob.size) throw new Error('The server returned an empty file. Please try again.');
    resultUrl = URL.createObjectURL(blob);
    const download = byId('download-link');
    download.href = resultUrl;
    download.download = filename;
    byId('open-link').href = resultUrl;

    badge.textContent = isDemo ? 'PRESET SAMPLE' : 'YOUR READING';
    badge.hidden = false;
    progress.value = 100;
    byId('output-subtitle').textContent = isDemo ? 'A sample reading, rendered with preset notes.' : 'Your annotated reading is ready.';
    showStatus(isDemo ? 'Sample ready. These are preset annotations on an original sample, rendered with the same PDF and handwriting pipeline. No Gemini request was made.' : 'Your PDF is ready. Review the generated notes and highlights against the original reading.');
    result.hidden = false;
    byId('preview-note').textContent = 'Preparing the first-page preview…';
    try {
      const payload = new FormData();
      payload.set('file', blob, filename);
      const previewResponse = await request('/api/preview', {method: 'POST', body: payload});
      if (!(previewResponse.headers.get('content-type') || '').includes('image/png')) throw new Error('Preview unavailable');
      const imageBlob = await previewResponse.blob();
      if (!imageBlob.size) throw new Error('Preview unavailable');
      previewUrl = URL.createObjectURL(imageBlob);
      const previewImage = byId('pdf-preview');
      previewImage.src = previewUrl;
      previewImage.alt = isDemo ? 'First page of the sample PDF with preset annotations, rendered by Auto Annotator' : 'First page of your completed annotated PDF';
      previewImage.hidden = false;
      byId('preview-note').textContent = 'First-page preview. Download the PDF to see every page.';
    } catch (_) {
      byId('pdf-preview').hidden = true;
      byId('preview-note').textContent = 'Your PDF is ready, but the page preview is unavailable. Download it to see every page.';
    }
  }

  async function runLocal(payload, filename) {
    payload.set('mode', regenerate.checked ? 'regenerate' : 'resume');
    const started = await (await request('/api/annotate', {method: 'POST', body: payload})).json();
    if (!started.job_id) throw new Error('The local server did not create a job.');
    const jobId = encodeURIComponent(started.job_id);
    const began = Date.now();
    while (Date.now() - began < 3600000) {
      const job = await (await request(`/api/status/${jobId}`)).json();
      if (job.total > 0) progress.value = Math.min(100, 100 * job.done / job.total);
      showStatus(`${job.message || 'Processing your reading…'}${job.total ? `\n${job.done} of ${job.total} reading pages generated.` : ''}`);
      if (job.status === 'failed') throw new Error(job.message || 'The local job failed.');
      if (job.status === 'complete') {
        const response = await request(`/api/download/${jobId}`);
        await displayPdf(response, filename, false);
        return;
      }
      await pause(1500);
    }
    throw new Error('The local job is still running. The app can reuse compatible cached annotations when you retry later.');
  }

  function validateFile(file) {
    if (!file) throw new Error('Choose a PDF reading first.');
    if (!/\.pdf$/i.test(file.name)) throw new Error('Choose a PDF file (.pdf).');
    if (file.size === 0) throw new Error('This file is empty. Choose a different PDF.');
    const limit = localMode() ? 500 * 1024 * 1024 : config.max_upload_bytes;
    if (file.size > limit) throw new Error(`Choose a PDF smaller than ${formatSize(limit)}${localMode() ? '.' : ', or use the local workflow for a larger reading.'}`);
  }

  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (busy || !config) return;
    let started = false;
    try {
      const file = fileInput.files[0];
      validateFile(file);
      const apiKey = keyInput.value.trim();
      if (!localMode() && !apiKey) throw new Error('Enter your Gemini API key, or try the sample without a key.');
      const payload = new FormData();
      payload.set('file', file);
      payload.set('api_key', apiKey);
      const useLocal = localMode();
      const filename = `${file.name.replace(/\.pdf$/i, '')}_Notes.pdf`;
      setBusy(true);
      started = true;
      if (useLocal) await runLocal(payload, filename);
      else await displayPdf(await request('/api/process', {method: 'POST', body: payload}), filename, false);
    } catch (error) {
      showStatus(error.message || 'The reading could not be processed.', true);
      if (started) byId('output-subtitle').textContent = 'Your reading needs another try.';
    } finally {
      if (started) setBusy(false);
    }
  });

  demoButton.addEventListener('click', async () => {
    if (busy || !config) return;
    setBusy(true, true);
    try {
      await displayPdf(await request('/api/demo', {method: 'POST'}), 'AutoAnnotator_Demo_Notes.pdf', true);
    } catch (error) {
      showStatus(error.message || 'The sample could not be rendered.', true);
      byId('output-subtitle').textContent = 'The sample needs another try.';
    } finally {
      setBusy(false);
    }
  });

  fileInput.addEventListener('change', showFile);
  fullLocal.addEventListener('change', updateMode);
  for (const eventName of ['dragenter', 'dragover']) dropZone.addEventListener(eventName, event => {
    event.preventDefault();
    if (!busy) dropZone.classList.add('dragging');
  });
  for (const eventName of ['dragleave', 'drop']) dropZone.addEventListener(eventName, event => {
    event.preventDefault();
    dropZone.classList.remove('dragging');
  });
  dropZone.addEventListener('drop', event => {
    if (busy || !config || !event.dataTransfer?.files.length) return;
    const files = event.dataTransfer.files;
    if (files.length !== 1) { showStatus('Choose one PDF at a time.', true); return; }
    const transfer = new DataTransfer();
    transfer.items.add(files[0]);
    fileInput.files = transfer.files;
    showFile();
  });

  async function initialize() {
    try {
      config = await (await request('/api/config')).json();
      if (!Number.isFinite(config.max_upload_bytes) || !Number.isFinite(config.max_pages)) throw new Error('The app returned an invalid configuration. Please reload.');
      byId('runtime-badge').textContent = config.hosted ? 'HOSTED EDITION' : 'LOCAL EDITION';
      byId('local-options').hidden = config.hosted;
      updateMode();
      setBusy(false);
    } catch (error) {
      config = null;
      byId('runtime-badge').textContent = 'UNAVAILABLE';
      showStatus(`${error.message || 'The app is unavailable.'} Reload the page to reconnect.`, true);
    }
  }
  initialize();
})();
