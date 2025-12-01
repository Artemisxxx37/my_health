// Application principale
class DiagnoXApp {
    constructor() {
        this.currentView = 'chat';
        this.init();
    }

    init() {
        this.setupNavigation();
        this.setupPredictionForm();
        this.requestNotificationPermission();
    }

    setupNavigation() {
        const navButtons = document.querySelectorAll('.nav-btn');
        
        navButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const view = btn.dataset.view;
                this.switchView(view);
                
                // Mettre à jour les boutons actifs
                navButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });
    }

    switchView(viewName) {
        // Cacher toutes les vues
        document.querySelectorAll('.view').forEach(view => {
            view.classList.remove('active');
        });

        // Afficher la vue sélectionnée
        const targetView = document.getElementById(`${viewName}-view`);
        if (targetView) {
            targetView.classList.add('active');
            this.currentView = viewName;

            // Charger les données si nécessaire
            if (viewName === 'history') {
                this.loadHistory();
            }
        }
    }

    async loadHistory() {
        const historyList = document.getElementById('history-list');
        historyList.innerHTML = '<p class="empty-state">Chargement...</p>';

        try {
            const consultations = await API.getHistory();
            
            if (consultations.length === 0) {
                historyList.innerHTML = '<p class="empty-state">Aucune consultation enregistrée</p>';
                return;
            }

            historyList.innerHTML = '';
            
            consultations.forEach(consultation => {
                const item = this.createHistoryItem(consultation);
                historyList.appendChild(item);
            });

        } catch (error) {
            console.error('Error loading history:', error);
            historyList.innerHTML = '<p class="empty-state">Erreur lors du chargement de l\'historique</p>';
        }
    }

    createHistoryItem(consultation) {
        const item = document.createElement('div');
        item.className = 'history-item';
        
        const date = new Date(consultation.timestamp);
        const dateStr = date.toLocaleDateString('fr-FR', {
            day: '2-digit',
            month: 'long',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });

        const symptomsText = consultation.symptoms && consultation.symptoms.length > 0
            ? consultation.symptoms.join(', ')
            : 'Symptômes non spécifiés';

        const diagnosisText = consultation.diagnosis && consultation.diagnosis.length > 0
            ? consultation.diagnosis[0].disease
            : 'Diagnostic en attente';

        item.innerHTML = `
            <div class="history-item-header">
                <strong>${diagnosisText}</strong>
                <span class="history-item-date">${dateStr}</span>
            </div>
            <div class="history-item-symptoms">
                Symptômes: ${symptomsText}
            </div>
        `;

        item.addEventListener('click', () => {
            this.showConsultationDetails(consultation);
        });

        return item;
    }

    showConsultationDetails(consultation) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Détails de la consultation</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <p><strong>Date:</strong> ${new Date(consultation.timestamp).toLocaleString('fr-FR')}</p>
                    <p><strong>Message:</strong> ${consultation.message}</p>
                    ${consultation.symptoms ? `<p><strong>Symptômes:</strong> ${consultation.symptoms.join(', ')}</p>` : ''}
                    ${consultation.diagnosis && consultation.diagnosis.length > 0 ? `
                        <div class="diagnosis-details">
                            <h4>Diagnostics possibles:</h4>
                            ${consultation.diagnosis.map(d => `
                                <div class="diagnosis-item">
                                    <strong>${d.disease}</strong> - ${d.confidence}% de confiance
                                    <p>Gravité: ${d.severity}</p>
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Fermer la modal
        modal.querySelector('.modal-close').addEventListener('click', () => {
            modal.remove();
        });

        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }

    setupPredictionForm() {
        const form = document.getElementById('prediction-form');
        
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.generatePrediction();
        });
    }

    async generatePrediction() {
        const age = document.getElementById('user-age').value;
        const gender = document.getElementById('user-gender').value;
        const lifestyle = document.getElementById('user-lifestyle').value;

        const userData = {
            age: parseInt(age) || undefined,
            gender: gender || undefined,
            lifestyle: lifestyle || undefined
        };

        // Afficher le loading
        this.showLoading();

        try {
            const prediction = await API.getPrediction(userData);
            this.hideLoading();
            this.displayPredictionResults(prediction);

        } catch (error) {
            console.error('Error generating prediction:', error);
            this.hideLoading();
            showNotification('Erreur lors de la génération de l\'analyse prédictive', 'error');
        }
    }

    displayPredictionResults(prediction) {
        const resultsContainer = document.getElementById('prediction-results');
        resultsContainer.style.display = 'block';

        if (!prediction.has_predictions) {
            resultsContainer.innerHTML = `
                <div class="info-box">
                    <h3>Données insuffisantes</h3>
                    <p>${prediction.message}</p>
                </div>
            `;
            return;
        }

        let html = `
            <h3>Analyse Prédictive de Santé</h3>
            <div class="prediction-summary">
                ${prediction.message}
            </div>
        `;

        if (prediction.predictions && prediction.predictions.length > 0) {
            html += `<div class="risk-cards">`;
            
            prediction.predictions.forEach(pred => {
                const riskClass = pred.risk_level === 'high' ? 'high' : pred.risk_level === 'low' ? 'low' : 'medium';
                html += `
                    <div class="risk-card ${riskClass}">
                        <h4>${pred.disease}</h4>
                        <p><strong>Risque:</strong> ${pred.risk_score}%</p>
                        <p><strong>Niveau:</strong> ${this.translateRiskLevel(pred.risk_level)}</p>
                        ${pred.recommendations ? `
                            <div class="recommendations">
                                <strong>Recommandations:</strong>
                                <ul>
                                    ${pred.recommendations.map(r => `<li>${r}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                    </div>
                `;
            });
            
            html += `</div>`;
        }

        if (prediction.next_checkup) {
            html += `
                <div class="next-checkup">
                    <h4>Prochain contrôle recommandé</h4>
                    <p>${prediction.next_checkup}</p>
                </div>
            `;
        }

        resultsContainer.innerHTML = html;
    }

    translateRiskLevel(level) {
        const translations = {
            'high': 'Élevé',
            'medium': 'Moyen',
            'low': 'Faible'
        };
        return translations[level] || level;
    }

    showLoading() {
        const overlay = document.getElementById('loading-overlay');
        overlay.style.display = 'flex';
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        overlay.style.display = 'none';
    }

    requestNotificationPermission() {
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission();
        }
    }
}

// Styles pour la modal
const modalStyles = document.createElement('style');
modalStyles.textContent = `
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
        animation: fadeIn 0.2s ease-out;
    }

    .modal-content {
        background: white;
        border-radius: 12px;
        max-width: 600px;
        width: 90%;
        max-height: 80vh;
        overflow-y: auto;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
        animation: slideUp 0.3s ease-out;
    }

    .modal-header {
        padding: 1.5rem;
        border-bottom: 1px solid #E5E7EB;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .modal-close {
        background: none;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        color: #6B7280;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .modal-close:hover {
        background: #F3F4F6;
    }

    .modal-body {
        padding: 1.5rem;
    }

    .modal-body p {
        margin-bottom: 1rem;
    }

    .diagnosis-details {
        margin-top: 1.5rem;
        padding: 1rem;
        background: #F9FAFB;
        border-radius: 8px;
    }

    .diagnosis-item {
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #E5E7EB;
    }

    .diagnosis-item:last-child {
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
    }

    @keyframes slideUp {
        from {
            transform: translateY(50px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }

    .info-box {
        padding: 1.5rem;
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
    }

    .prediction-summary {
        padding: 1rem;
        background: #F9FAFB;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        white-space: pre-wrap;
    }

    .risk-cards {
        display: grid;
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .next-checkup {
        padding: 1rem;
        background: #DBEAFE;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        margin-top: 1.5rem;
    }

    .recommendations ul {
        margin-left: 1.5rem;
        margin-top: 0.5rem;
    }

    .recommendations li {
        margin-bottom: 0.25rem;
    }
`;
document.head.appendChild(modalStyles);

// Initialiser l'application
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new DiagnoXApp();
});
