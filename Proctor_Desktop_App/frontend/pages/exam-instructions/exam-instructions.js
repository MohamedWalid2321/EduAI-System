'use strict';

document.addEventListener('DOMContentLoaded', async () => {
  // ── T012 / US3 — Session Guard ───────────────────────────────────────────
  // Must run before any DOM queries; redirects immediately on missing session.
  const sessionResult = await window.bridge.getExamSession();
  if (!sessionResult?.ok || !sessionResult?.session?.attemptId) {
    window.location.replace('../login/index.html');
    return;
  }

  // ── T009 / US1 — Element queries ─────────────────────────────────────────
  const agreeCheckbox = document.getElementById('agreeCheckbox');
  const startBtn      = document.getElementById('startBtn');
  const btnText       = document.getElementById('btnText');   // T010 / US2
  const btnSpinner    = document.getElementById('btnSpinner'); // T010 / US2

  // ── T009 / US1 — Checkbox gates the Start Exam button ────────────────────
  agreeCheckbox.addEventListener('change', () => {
    startBtn.disabled = !agreeCheckbox.checked;
  });

  // ── T011 / US2 — Start Exam loading state + navigation ───────────────────
  startBtn.addEventListener('click', async () => {
    if (startBtn.disabled) return;

    // Enter loading state
    startBtn.disabled = true;
    btnText.textContent = 'Starting…';
    btnSpinner.hidden = false;
    btnSpinner.setAttribute('aria-hidden', 'false');

    // Activate fullscreen/kiosk lockdown before entering the exam
    await window.bridge.startLockdown();

    // Navigate to the active exam page
    window.location.href = '../exam/index.html';
  });
});
