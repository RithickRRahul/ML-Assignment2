document.addEventListener('DOMContentLoaded', () => {
    // --- Get DOM Elements ---
    const form = document.getElementById('prediction-form');
    const partySelect = document.getElementById('party');
    const genderSelect = document.getElementById('gender');
    const yearInput = document.getElementById('year');
    const electorsInput = document.getElementById('electors');
    const totvotpollInput = document.getElementById('totvotpoll');
    const voteShareDisplay = document.getElementById('vote-share-output');
    const predictButton = document.getElementById('predict-button');
    const loadingSpinner = document.getElementById('loading-spinner');

    const resultsSection = document.getElementById('results-section');
    const predictionOutput = document.getElementById('prediction-output');
    const winProbOutput = document.getElementById('win-probability-output');
    const comparisonOutput = document.getElementById('comparison-output');

    // --- Historical Data (Only needed for comparison now) ---
    const partyGenderWinRates = {
        "BJP_M": 60.2, "BJP_F": 55.1, "BJP_O": 48.3,
        "INC_M": 44.5, "INC_F": 38.2, "INC_O": 30.1,
    };

    // --- FIXED MODEL ---
    const fixedModelName = "Gradient Boosting"; // Use GB by default

    // --- Calculate and Display Vote Share ---
    function calculateAndDisplayVoteShare() {
        const electors = parseFloat(electorsInput.value) || 0;
        const totvotpoll = parseFloat(totvotpollInput.value) || 0;
        let share = 0;
        if (electors > 0) {
            share = (totvotpoll / electors * 100);
        }
        voteShareDisplay.textContent = `${share.toFixed(2)}%`;
    }

    // Initial calculation and listeners for updates
    calculateAndDisplayVoteShare();
    electorsInput.addEventListener('input', calculateAndDisplayVoteShare);
    totvotpollInput.addEventListener('input', calculateAndDisplayVoteShare);

    // --- Form Submission Handler ---
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        loadingSpinner.style.display = 'block';
        resultsSection.style.display = 'none';
        predictionOutput.textContent = '';
        winProbOutput.textContent = '';
        comparisonOutput.innerHTML = '';
        predictionOutput.className = 'prediction-text';

        const electorsVal = parseFloat(electorsInput.value);
        const totvotpollVal = parseFloat(totvotpollInput.value);

        const payload = {
            party: partySelect.value,
            gender: genderSelect.value,
            year: parseInt(yearInput.value),
            electors: electorsVal,
            totvotpoll: totvotpollVal,
            selected_model: fixedModelName // Send fixed model name
        };

        // --- Make API Call ---
        const apiUrl = 'http://127.0.0.1:5000/predict'; // UPDATE THIS URL

        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            loadingSpinner.style.display = 'none';
            resultsSection.style.display = 'block';

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Unknown server error' }));
                throw new Error(`API Error (${response.status}): ${errorData.error || response.statusText}`);
            }

            const result = await response.json();

            // --- Display Results ---
            if (result.prediction === 1) {
                predictionOutput.textContent = '✅ PREDICTION: LIKELY TO WIN';
                predictionOutput.classList.add('success');
            } else {
                predictionOutput.textContent = '❌ PREDICTION: LIKELY TO LOSE';
                 predictionOutput.classList.add('error');
            }

            if (result.win_probability !== null && result.win_probability !== undefined) {
                 winProbOutput.textContent = `Win Probability: ${(result.win_probability * 100).toFixed(1)}%`;
            } else {
                 winProbOutput.textContent = 'Win Probability: N/A';
            }

            // Comparison Stats (using pre-defined JS object)
            const combinedKey = `${payload.party}_${payload.gender}`;
            const histRateData = partyGenderWinRates[combinedKey];

            if (histRateData !== null && histRateData !== undefined) {
                 const historical = histRateData / 100.0;
                 const model_prob = result.win_probability !== null ? result.win_probability : (result.prediction === 1 ? 1.0 : 0.0);
                 const difference = (model_prob - historical) * 100;

                comparisonOutput.innerHTML = `
                    <p><strong>Comparison (Party/Gender Only):</strong></p>
                    <p>- Historical win rate: ${histRateData.toFixed(1)}%</p>
                    <p>- Model prediction probability: ${(model_prob * 100).toFixed(1)}%</p>
                    <p>- Difference: ${difference.toFixed(1)}%</p>
                `;
            } else {
                 comparisonOutput.innerHTML = '<p><em>(No specific historical win rate for this Party/Gender combination)</em></p>';
            }


        } catch (error) {
            loadingSpinner.style.display = 'none';
            resultsSection.style.display = 'block';
             predictionOutput.textContent = `⚠️ Error: ${error.message}`;
             predictionOutput.classList.add('error');
            console.error('Prediction API call failed:', error);
        }
    });
});