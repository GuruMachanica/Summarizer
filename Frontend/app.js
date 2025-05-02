// Dark mode toggle logic for Uiverse.io switch
// This ensures the theme is toggled and preference is saved in localStorage
document.addEventListener('DOMContentLoaded', function() {
  const toggle = document.getElementById('toggle');
  if (!toggle) return; // Exit if toggle not found
  // Set initial state from localStorage
  if (localStorage.getItem('summarizer-dark') === '1') {
    toggle.checked = true;
    document.body.classList.add('dark');
  } else {
    toggle.checked = false;
    document.body.classList.remove('dark');
  }
  // Listen for toggle changes
  toggle.addEventListener('change', function() {
    document.body.classList.toggle('dark', this.checked);
    localStorage.setItem('summarizer-dark', this.checked ? '1' : '0');
  });
});

// Loader logic for Summarize button
// Handles UI feedback during async summarization
const summarizeBtn = document.getElementById('summarize-btn');
const loader = document.getElementById('loader');
const btnText = document.getElementById('btn-text');

// Form submission handler for summarization
// Sends user input to backend and displays results
document.getElementById('summarize-form').addEventListener('submit', async function(e) {
  e.preventDefault();
  document.getElementById('result').style.display = 'none';
  document.getElementById('error').textContent = '';

  // Show loader
  loader.style.display = 'inline-block';
  btnText.textContent = 'Summarizing...';
  summarizeBtn.disabled = true;

  // Collect form values
  const text = document.getElementById('text').value;
  const url = document.getElementById('url').value;
  const method = document.getElementById('method').value;
  const ratio = parseFloat(document.getElementById('ratio').value);
  const min_len = parseInt(document.getElementById('min_len').value);
  const min_abs = parseInt(document.getElementById('min_abs').value);
  const max_abs = parseInt(document.getElementById('max_abs').value);
  const top_k = parseInt(document.getElementById('top_k').value);
  const abstractive = method === 'abstractive';
  const language = document.getElementById('language')?.value;

  // Prepare payload for API
  const payload = {
    text,
    url,
    method,
    ratio,
    min_len,
    min_abs,
    max_abs,
    top_k,
    abstractive,
    language
  };

  // Removed language-related logic
  delete payload.language;

  try {
    // Send POST request to backend API
    const res = await fetch('/api/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error('Summarization failed.');
    const data = await res.json();
    // Display results
    document.getElementById('summary').textContent = data.summary;
    document.getElementById('evaluation').textContent = JSON.stringify(data.evaluation, null, 2);
    document.getElementById('rouge').textContent = JSON.stringify(data.rouge, null, 2);
    document.getElementById('keywords').textContent = data.keywords.join(', ');
    document.getElementById('result').style.display = 'block';
  } catch (err) {
    // Show error message
    document.getElementById('error').textContent = err.message;
  } finally {
    // Reset loader and button
    loader.style.display = 'none';
    btnText.textContent = 'Summarize';
    summarizeBtn.disabled = false;
  }
});
