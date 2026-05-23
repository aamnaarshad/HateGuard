// static/script.js

const MIN_WORDS = 3;   // minimum word count to allow analysis

const textInput      = document.getElementById('textInput');
const charCount      = document.getElementById('charCount');
const wordHint       = document.getElementById('wordHint');
const analyzeBtn     = document.getElementById('analyzeBtn');
const clearBtn       = document.getElementById('clearBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingState   = document.getElementById('loadingState');
const errorState     = document.getElementById('errorState');
const resultsContent = document.getElementById('resultsContent');
const cleanedText    = document.getElementById('cleanedText');
const cardsGrid      = document.getElementById('cardsGrid');
const verdictBanner  = document.getElementById('verdictBanner');
const verdictText    = document.getElementById('verdictText');

// ── Word count helper ────────────────────────────────
function countWords(text) {
  return text.trim().split(/\s+/).filter(w => w.length > 0).length;
}

// ── Live character + word count feedback ─────────────
textInput.addEventListener('input', () => {
  const val   = textInput.value;
  const chars = val.length;
  const words = countWords(val);

  charCount.textContent = chars;

  if (chars === 0) {
    wordHint.textContent = `Minimum ${MIN_WORDS} words required`;
    wordHint.className   = 'word-hint';
  } else if (words < MIN_WORDS) {
    wordHint.textContent = `${words} word${words === 1 ? '' : 's'} — need at least ${MIN_WORDS}`;
    wordHint.className   = 'word-hint warn';
  } else {
    wordHint.textContent = `${words} words ✓`;
    wordHint.className   = 'word-hint';
  }
});

// ── Example chips ────────────────────────────────────
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    textInput.value = chip.dataset.text;
    textInput.dispatchEvent(new Event('input'));  // trigger word counter
    textInput.focus();
  });
});

// ── Clear ────────────────────────────────────────────
clearBtn.addEventListener('click', () => {
  textInput.value       = '';
  charCount.textContent = '0';
  wordHint.textContent  = `Minimum ${MIN_WORDS} words required`;
  wordHint.className    = 'word-hint';
  hideResults();
  textInput.focus();
});

// ── Analyze ──────────────────────────────────────────
analyzeBtn.addEventListener('click', runAnalysis);

textInput.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') runAnalysis();
});

// ── Main analysis ────────────────────────────────────
async function runAnalysis() {
  const text  = textInput.value.trim();
  const words = countWords(text);

  // Empty check
  if (!text) {
    showError('Please enter some text before analyzing.');
    return;
  }

  // Minimum word check — the key new validation
  if (words < MIN_WORDS) {
    showError(
      `Please enter at least ${MIN_WORDS} words. ` +
      `A single word has no context for the model to classify reliably.`
    );
    return;
  }

  showLoading();

  try {
    const response = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text })
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.error || 'Something went wrong. Please try again.');
      return;
    }

    displayResults(data);

  } catch (err) {
    showError('Could not connect to the server. Make sure Flask is running.');
  }
}

// ── Display results ──────────────────────────────────
function displayResults(data) {
  cleanedText.textContent = data.cleaned_text || '(empty after cleaning)';

  cardsGrid.innerHTML = '';

  data.results.forEach((result, index) => {
    const cls  = labelToCssClass(result.label);
    const card = document.createElement('div');
    card.className          = `model-card ${cls}`;
    card.style.animationDelay = `${index * 0.08}s`;

    card.innerHTML = `
      <div class="card-model-name">${result.model}</div>
      <div class="card-label ${cls}">${result.label}</div>
      <div class="conf-bar-bg">
        <div class="conf-bar-fill ${cls}" data-width="${result.confidence}"></div>
      </div>
      <div class="conf-value">${result.confidence}% confidence</div>
    `;

    cardsGrid.appendChild(card);
  });

  // Animate bars after DOM paint
  setTimeout(() => {
    document.querySelectorAll('.conf-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.width + '%';
    });
  }, 80);

  const verdict      = getMajorityVote(data.results);
  const verdictClass = labelToCssClass(verdict);

  verdictText.textContent = verdict;
  verdictText.className   = `verdict-text ${verdictClass}`;
  verdictBanner.className = `verdict-banner ${verdictClass}`;

  hideLoading();
  resultsContent.classList.add('active');
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Helpers ──────────────────────────────────────────
function getMajorityVote(results) {
  const counts = {};
  results.forEach(r => { counts[r.label] = (counts[r.label] || 0) + 1; });
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
}

function labelToCssClass(label) {
  if (label === 'Hate Speech')        return 'hate';
  if (label === 'Offensive Language') return 'offensive';
  return 'neither';
}

function showLoading() {
  analyzeBtn.disabled = true;
  loadingState.classList.add('active');
  errorState.classList.remove('active');
  resultsContent.classList.remove('active');
  resultsSection.style.display = 'block';
}

function hideLoading() {
  analyzeBtn.disabled = false;
  loadingState.classList.remove('active');
}

function showError(message) {
  analyzeBtn.disabled = false;
  loadingState.classList.remove('active');
  resultsContent.classList.remove('active');
  errorState.textContent = '⚠ ' + message;
  errorState.classList.add('active');
  resultsSection.style.display = 'block';
}

function hideResults() {
  loadingState.classList.remove('active');
  errorState.classList.remove('active');
  resultsContent.classList.remove('active');
  resultsSection.style.display = 'none';
}

// Also update app.py backend — the MIN_WORDS check is in JS
// but we match it server-side too (see app.py)
hideResults();