'use strict';

/**
 * report.js
 * Controller for the Session Report page.
 * Loads session logs from disk, parses events, and renders the timeline.
 */

class ReportController {
  constructor() {
    this.sessionId = null;
    this.events = [];
    this.filter = 'all';

    // DOM Elements
    this.examTitle = document.getElementById('examTitle');
    this.sessionIdDisplay = document.getElementById('sessionIdDisplay');
    this.peakRiskScore = document.getElementById('peakRiskScore');
    this.totalAlerts = document.getElementById('totalAlerts');
    this.sessionDuration = document.getElementById('sessionDuration');
    this.violationSummary = document.getElementById('violationSummary');
    this.eventTimeline = document.getElementById('eventTimeline');
    this.exportBtn = document.getElementById('exportPdfBtn');
    this.backBtn = document.getElementById('backBtn');
    this.toast = document.getElementById('toast');

    this.init();
  }

  async init() {
    // 1. Identify session (from URL params or stored result)
    const params = new URLSearchParams(window.location.search);
    this.sessionId = params.get('sessionId');

    if (!this.sessionId) {
      const result = await window.bridge.getExamSession();
      if (result.ok) {
        this.sessionId = result.session.attemptId;
        this.examTitle.textContent = result.session.quizTitle;
      }
    }

    if (!this.sessionId) {
      this.eventTimeline.innerHTML = '<div class="error-state">No session ID provided.</div>';
      return;
    }

    this.sessionIdDisplay.textContent = `Session: ${this.sessionId}`;

    // 2. Load log
    await this.loadLog();

    // 3. Wire buttons
    this.exportBtn.addEventListener('click', () => this.exportPdf());
    this.backBtn.addEventListener('click', () => window.location.href = '../result/index.html');

    // 4. Wire filters
    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelector('.filter-btn.active').classList.remove('active');
        e.target.classList.add('active');
        this.filter = e.target.dataset.filter;
        this.renderTimeline();
      });
    });
  }

  async loadLog() {
    const result = await window.bridge.getSessionLog(this.sessionId);
    if (!result.ok) {
      this.eventTimeline.innerHTML = `<div class="error-state">Error loading log: ${result.error.message}</div>`;
      return;
    }

    // Parse JSONL
    this.events = result.data.map(line => {
      try {
        return JSON.parse(line);
      } catch (e) {
        return null;
      }
    }).filter(e => e !== null);

    this.calculateSummary();
    this.renderTimeline();
  }

  calculateSummary() {
    let peak = 0;
    let alerts = 0;
    const violations = {};

    this.events.forEach(e => {
      if (e.type === 'riskScore') {
        peak = Math.max(peak, e.score);
      }
      if (e.type === 'alert') {
        alerts++;
        const service = this.formatServiceName(e.code);
        violations[service] = (violations[service] || 0) + 1;
      }
    });

    this.peakRiskScore.textContent = peak;
    this.totalAlerts.textContent = alerts;

    // Calculate duration
    if (this.events.length >= 2) {
      const start = new Date(this.events[0].timestamp || Date.now());
      const end = new Date(this.events[this.events.length - 1].timestamp || Date.now());
      const diffMs = end - start;
      const mins = Math.floor(diffMs / 60000);
      const secs = Math.floor((diffMs % 60000) / 1000);
      this.sessionDuration.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    // Render violation tags
    this.violationSummary.innerHTML = Object.entries(violations).map(([service, count]) => `
      <div class="violation-tag">
        <span class="label">${service}</span>
        <span class="count">${count}</span>
      </div>
    `).join('') || '<div class="alert-placeholder">No violations recorded.</div>';
  }

  renderTimeline() {
    const filtered = this.filter === 'all' 
      ? this.events 
      : this.events.filter(e => e.type === 'alert');

    this.eventTimeline.innerHTML = filtered.map(e => this.createTimelineItem(e)).join('');
  }

  createTimelineItem(e) {
    const isAlert = e.type === 'alert';
    const time = new Date(e.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    let typeLabel = e.type;
    if (e.service) typeLabel = this.formatServiceName(e.service);
    if (e.code) typeLabel = this.formatServiceName(e.code);

    let body = '';
    if (e.type === 'alert') body = e.message;
    else if (e.type === 'riskScore') body = `Risk score updated to ${e.score}`;
    else if (e.service) {
      const p = e.payload || {};
      if (e.service === 'eye-gaze') body = `Gaze status: ${p.status}`;
      else if (e.service === 'speech-detection') body = `Speech detected: ${p.is_speech_detected}`;
      else if (e.service === 'object-detection') body = `Detected: ${p.objects?.join(', ') || 'None'}`;
      else body = 'System detection event';
    }

    return `
      <div class="timeline-item ${isAlert ? 'is-alert' : ''} severity-${e.severity || 'none'}">
        <div class="timeline-dot"></div>
        <div class="event-card">
          <div class="event-header">
            <span class="event-type">${typeLabel}</span>
            <span class="event-time">${time}</span>
          </div>
          <div class="event-body">${body}</div>
        </div>
      </div>
    `;
  }

  formatServiceName(codeOrService) {
    const s = codeOrService.toLowerCase();
    if (s.includes('gaze')) return 'Eye Tracking';
    if (s.includes('face')) return 'Face Recognition';
    if (s.includes('object')) return 'Object Detection';
    if (s.includes('speech')) return 'Speech Detection';
    return codeOrService;
  }

  async exportPdf() {
    this.showToast('Generating PDF...');
    const filename = `Lumina_Report_${this.sessionId}.pdf`;
    const result = await window.bridge.exportPdf(filename);
    
    if (result.ok) {
      this.showToast('Report saved successfully!');
    }
  }

  showToast(msg) {
    this.toast.textContent = msg;
    this.toast.classList.remove('hidden');
    setTimeout(() => this.toast.classList.add('hidden'), 3000);
  }
}

new ReportController();
