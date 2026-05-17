// static/script.js
// --------------------------------------------------
// Handles all UI interactions:
//   - Sending text to the /predict route
//   - Displaying results without reloading the page
// --------------------------------------------------

// ── Grab DOM elements we need ────────────────────────
const textInput      = document.getElementById('textInput');
const charCount      = document.getElementById('charCount');
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


// ── Character counter ────────────────────────────────
textInput.addEventListener('input', () => {
  charCount.textContent = textInput.value.length;
});


// ── Example chips ────────────────────────────────────
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    textInput.value = chip.dataset.text;
    charCount.textContent = textInput.value.length;
    textInput.focus();
  });
});


// ── Clear button ─────────────────────────────────────
clearBtn.addEventListener('click', () => {
  textInput.value = '';
  charCount.textContent = '0';
  hideResults();
  textInput.focus();
});


// ── Analyze button ───────────────────────────────────
analyzeBtn.addEventListener('click', runAnalysis);

// Also allow pressing Enter + Ctrl/Cmd to submit
textInput.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    runAnalysis();
  }
});


// ── Main analysis function ───────────────────────────
async function runAnalysis() {
  const text = textInput.value.trim();

  // Don't submit if empty
  if (!text) {
    showError('Please enter some text before analyzing.');
    return;
  }

  // Show loading, hide previous results
  showLoading();

  try {
    // Send POST request to Flask backend
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text })
    });

    const data = await response.json();

    // If Flask returned an error (like empty text)
    if (!response.ok) {
      showError(data.error || 'Something went wrong. Please try again.');
      return;
    }

    // All good — display the results
    displayResults(data);

  } catch (err) {
    // Network error or server crash
    showError('Could not connect to the server. Make sure Flask is running.');
  }
}


// ── Display results ──────────────────────────────────
function displayResults(data) {
  // Show the cleaned/preprocessed text
  cleanedText.textContent = data.cleaned_text || '(empty after cleaning)';

  // Build the three model cards
  cardsGrid.innerHTML = '';  // clear old cards

  data.results.forEach((result, index) => {
    const cssClass = labelToCssClass(result.label);

    const card = document.createElement('div');
    card.className = `model-card ${cssClass}`;
    card.style.animationDelay = `${index * 0.08}s`;

    card.innerHTML = `
      <div class="card-model-name">${result.model}</div>
      <div class="card-label ${cssClass}">${result.label}</div>
      <div class="conf-bar-bg">
        <div class="conf-bar-fill ${cssClass}" data-width="${result.confidence}"></div>
      </div>
      <div class="conf-value">${result.confidence}% confidence</div>
    `;

    cardsGrid.appendChild(card);
  });

  // Animate the confidence bars (slight delay lets DOM paint first)
  setTimeout(() => {
    document.querySelectorAll('.conf-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.width + '%';
    });
  }, 100);

  // Calculate the majority verdict (most common label among 3 models)
  const verdict = getMajorityVote(data.results);
  const verdictClass = labelToCssClass(verdict);

  verdictText.textContent = verdict;
  verdictText.className = `verdict-text ${verdictClass}`;
  verdictBanner.className = `verdict-banner ${verdictClass}`;

  // Show results, hide loading
  hideLoading();
  resultsContent.classList.add('active');

  // Smooth scroll to results
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}


// ── Majority vote helper ─────────────────────────────
function getMajorityVote(results) {
  // Count how many models voted for each label
  const counts = {};
  results.forEach(r => {
    counts[r.label] = (counts[r.label] || 0) + 1;
  });

  // Return the label with the highest count
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    [0][0];
}


// ── CSS class helper ─────────────────────────────────
function labelToCssClass(label) {
  if (label === 'Hate Speech')        return 'hate';
  if (label === 'Offensive Language') return 'offensive';
  return 'neither';
}


// ── UI state helpers ─────────────────────────────────
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

// Hide results section initially
hideResults();
