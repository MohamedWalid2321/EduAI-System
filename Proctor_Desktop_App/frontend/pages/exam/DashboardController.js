'use strict';

/**
 * DashboardController.js
 * Manages the live proctoring dashboard in the exam sidebar.
 * Listens for AI events via bridge.onAiEvent and updates the UI.
 */
class DashboardController {
  constructor() {
    this.riskScore = 0;
    this.alerts = [];
    this.maxAlerts = 20;
    this.statusPollIntervalId = null;

    // DOM Elements
    this.statusPills = {
      'face-recognition': document.querySelector('.status-row[data-service="face-recognition"] .status-indicator'),
      'eye-gaze': document.querySelector('.status-row[data-service="eye-gaze"] .status-indicator'),
      'object-detection': document.querySelector('.status-row[data-service="object-detection"] .status-indicator'),
      'speech-detection': document.querySelector('.status-row[data-service="speech-detection"] .status-indicator')
    };

    this.riskValue = document.getElementById('riskValue');
    this.gaugeNeedle = document.getElementById('gaugeNeedle');
    this.alertFeed = document.getElementById('alertFeed');
    
    // SVG Gauge constants
    this.gaugeRadius = 40;
    this.gaugeCenter = { x: 50, y: 50 };
  }

  init() {
    // Subscribe to AI events from bridge
    this.unsubscribe = window.bridge.onAiEvent((event) => {
      this.handleAiEvent(event);
    });

    // Status should reflect service lifecycle, not only detection traffic.
    this.syncServiceStatuses();
    this.statusPollIntervalId = setInterval(() => {
      this.syncServiceStatuses();
    }, 3000);
  }

  destroy() {
    if (this.unsubscribe) this.unsubscribe();
    if (this.statusPollIntervalId) {
      clearInterval(this.statusPollIntervalId);
      this.statusPollIntervalId = null;
    }
  }

  async syncServiceStatuses() {
    if (!window.bridge?.aiRpc) return;

    try {
      const response = await window.bridge.aiRpc('queryStatus', {}, { timeoutMs: 5000 });
      if (!response?.ok || !response.result) return;

      Object.entries(response.result).forEach(([service, status]) => {
        if (status === 'running') {
          this.updateServiceStatus(service, 'active');
        } else if (status === 'stopped') {
          this.updateServiceStatus(service, 'inactive');
        }
      });
    } catch {
      // Ignore transient bridge/router errors; next poll will retry.
    }
  }

  handleAiEvent(event) {
    const { type, method, params } = event;

    // Notification type can be direct (from orchestrator) or wrapped in method (from router)
    const msgType = type || method;
    const data = params || event;

    switch (msgType) {
      case 'detection':
        this.updateServiceStatus(data.service, 'active');
        break;
      case 'riskScore':
        this.updateRiskScore(data.score);
        break;
      case 'alert':
        this.addAlert(data);
        break;
      case 'serviceError':
        this.updateServiceStatus(data.service, 'error');
        break;
    }
  }

  updateServiceStatus(service, status) {
    const pill = this.statusPills[service];
    if (!pill) return;

    pill.className = `status-indicator ${status}`;
    pill.textContent = status.charAt(0).toUpperCase() + status.slice(1);
  }

  updateRiskScore(score) {
    this.riskScore = score;
    this.riskValue.textContent = score;

    // Update SVG Gauge path
    // Math: arc from 180deg (left) to 0deg (right)
    // 0 score = 180deg, 100 score = 0deg
    const angle = 180 - (score * 1.8);
    const rad = (angle * Math.PI) / 180;
    
    const x = this.gaugeCenter.x + this.gaugeRadius * Math.cos(rad);
    const y = this.gaugeCenter.y - this.gaugeRadius * Math.sin(rad);

    // Update the gauge-fill path
    // D string: Move to start (10, 50), then draw arc to calculated (x, y)
    this.gaugeNeedle.setAttribute('d', `M 10 50 A 40 40 0 0 1 ${x.toFixed(2)} ${y.toFixed(2)}`);

    // Update color based on score
    let color = 'var(--color-primary)';
    if (score > 40) color = 'var(--color-tertiary)';
    if (score > 70) color = 'var(--color-error)';
    this.gaugeNeedle.style.stroke = color;
  }

  addAlert(alert) {
    // Remove placeholder if present
    const placeholder = this.alertFeed.querySelector('.alert-placeholder');
    if (placeholder) placeholder.remove();

    // Create alert card
    const card = document.createElement('div');
    card.className = `alert-card severity-${alert.severity}`;
    
    const time = new Date(alert.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    card.innerHTML = `
      <div class="alert-header">
        <span>${this.formatServiceName(alert.code)}</span>
        <span class="alert-time">${time}</span>
      </div>
      <div class="alert-msg">${alert.message}</div>
    `;

    // Prepend to feed
    this.alertFeed.insertBefore(card, this.alertFeed.firstChild);

    // Maintain circular buffer
    if (this.alertFeed.children.length > this.maxAlerts) {
      this.alertFeed.removeChild(this.alertFeed.lastChild);
    }
  }

  formatServiceName(code) {
    // Convert GAZE_OFF_SCREEN to "Gaze Tracking", etc.
    if (code.startsWith('GAZE')) return 'Eye Tracking';
    if (code.startsWith('NO_FACE')) return 'Face Recognition';
    if (code.startsWith('UNAUTHORIZED')) return 'Object Detection';
    if (code.startsWith('SPEECH')) return 'Speech Detection';
    return 'Proctoring';
  }
}

// Export for use in exam.js
window.DashboardController = DashboardController;
