'use strict';

// FR-002: Pass threshold — hardcoded 50% inclusive, no server-provided value expected.
const PASS_THRESHOLD = 50;

// Module-level result store set after successful IPC retrieval.
let resultData = null;

// ---------------------------------------------------------------------------
// Render helpers (T011 / T013)
// ---------------------------------------------------------------------------

/**
 * Animate the SVG circular progress ring to the given percentage.
 * T020 — depends on T018 SVG structure (r=88, .progress-circle class).
 * @param {number} pct  0–100
 */
function renderProgress(pct) {
  const r = 88;
  const circumference = 2 * Math.PI * r;
  const circle = document.getElementById('progress-ring').querySelector('.progress-circle');
  circle.style.strokeDasharray = circumference;
  circle.style.strokeDashoffset = circumference * (1 - pct / 100);
  document.getElementById('percentage-text').textContent = `${pct.toFixed(1)}%`;
}

/**
 * Populate the score summary hero card from a SubmitResult object.
 * T022 — uses new DOM IDs from T018.
 * @param {{ quizTitle: string, quizCode: string, score: number, totalQuestions: number, percentage: number }} data
 */
function renderScore(data) {
  document.getElementById('exam-title').textContent = data.quizTitle;
  document.getElementById('exam-code-chip').textContent = data.quizCode;
  document.getElementById('score-text').textContent = `${data.score} / ${data.totalQuestions} Score`;

  const chip = document.getElementById('pass-fail-chip');
  const passed = data.percentage >= PASS_THRESHOLD;
  chip.textContent = passed ? 'Passed' : 'Failed';
  chip.classList.remove('pass', 'fail');
  chip.classList.add(passed ? 'pass' : 'fail');

  renderProgress(data.percentage);
}

/**
 * Populate the per-question breakdown list with Phase 7 card markup.
 * T022 — section is always visible; data-correct drives filter tabs.
 * @param {Array<{ questionText: string, studentChoice: string, correctChoice: string, isCorrect: boolean }>} questions
 */
function renderBreakdown(questions) {
  if (!questions || questions.length === 0) {
    // FR-008: section stays empty; breakdown-section is always present in new layout.
    return;
  }

  const list = document.getElementById('breakdown-list');
  const fragment = document.createDocumentFragment();

  questions.forEach((q) => {
    const li = document.createElement('li');
    li.className = `question-card ${q.isCorrect ? 'correct' : 'incorrect'}`;
    li.dataset.correct = String(q.isCorrect);

    const yourAnswerClass = `answer-box your-answer${q.isCorrect ? '' : ' incorrect'}`;
    const yourAnswerLabel = q.isCorrect ? 'Your Selection' : 'Your Selection (Incorrect)';

    li.innerHTML = `
      <div class="q-header">
        <div class="q-icon ${q.isCorrect ? 'correct' : 'incorrect'}">${q.isCorrect ? '&#10003;' : '&#10007;'}</div>
        <p class="q-text">${escapeHtml(q.questionText)}</p>
      </div>
      <div class="answer-boxes">
        <div class="${yourAnswerClass}">
          <p class="answer-box-label">${yourAnswerLabel}</p>
          <p class="answer-box-value">${escapeHtml(q.studentChoice)}</p>
        </div>
        <div class="answer-box correct-answer">
          <p class="answer-box-label">Correct Answer</p>
          <p class="answer-box-value">${escapeHtml(q.correctChoice)}</p>
        </div>
      </div>
    `;

    fragment.appendChild(li);
  });

  list.appendChild(fragment);
}

/**
 * Wire up the All / Correct / Incorrect filter tabs.
 * T021 — populates counts and attaches click handlers.
 * @param {Array<{ isCorrect: boolean }>} questions
 */
function initFilterTabs(questions) {
  document.getElementById('count-all').textContent = questions.length;
  document.getElementById('count-correct').textContent = questions.filter((q) => q.isCorrect).length;
  document.getElementById('count-incorrect').textContent = questions.filter((q) => !q.isCorrect).length;

  const tabs = document.querySelectorAll('.filter-tab');
  const items = document.querySelectorAll('#breakdown-list li');

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');

      const filter = tab.dataset.filter;
      items.forEach((item) => {
        if (filter === 'all') {
          item.removeAttribute('hidden');
        } else if (filter === 'correct') {
          item.dataset.correct === 'true' ? item.removeAttribute('hidden') : item.setAttribute('hidden', '');
        } else {
          item.dataset.correct === 'false' ? item.removeAttribute('hidden') : item.setAttribute('hidden', '');
        }
      });
    });
  });

  // Activate "All" tab by default.
  const allTab = document.querySelector('.filter-tab[data-filter="all"]');
  if (allTab) {
    allTab.classList.add('active');
  }
}

/**
 * Minimal HTML escaper to prevent XSS from LMS-supplied strings.
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ---------------------------------------------------------------------------
// Entry point (T009)
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', async () => {
  const backBtn = document.getElementById('back-btn');

  // (1) Disable Back to Home while async data loading is in-flight (FR-004 C1).
  // back-btn is an <a> anchor, so use aria-disabled + pointer-events via a CSS class.
  backBtn.setAttribute('aria-disabled', 'true');
  backBtn.style.pointerEvents = 'none';
  backBtn.style.opacity = '0.4';

  // (2) Try the fast path — submitResult already cached in main.js.
  let res = await window.bridge.getSubmitResult();

  // (3) Fall through to recovery path if submitResult is absent (FR-004).
  if (!res.ok) {
    res = await window.bridge.getResult();

    if (!res.ok) {
      // No attemptId, 401, 5xx, network error — all branches redirect to Login.
      window.location.replace('../login/index.html');
      return;
    }
  }

  // (4) Data is available — store, re-enable navigation, render.
  resultData = res.data;
  backBtn.removeAttribute('aria-disabled');
  backBtn.style.pointerEvents = '';
  backBtn.style.opacity = '';

  // T022: renderScore uses new hero-card IDs including SVG ring.
  renderScore(resultData);

  // T022: renderBreakdown uses new question-card markup with data-correct.
  renderBreakdown(resultData.questions);

  // T022 (T021): Wire filter tabs after breakdown items exist in the DOM.
  initFilterTabs(resultData.questions ?? []);

  // T022 (US3): back-btn is now a <a> anchor in the page header.
  document.getElementById('back-btn').addEventListener('click', (e) => {
    e.preventDefault();
    window.bridge.clearSubmitResult();
  });
});
