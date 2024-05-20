document.getElementById('check_button').addEventListener('click', function() {
    // Show loading animation and change button text
    this.classList.add('loading');
    this.textContent = 'Predicting...';

    const post = document.getElementById('post').value;
    const replyCount = document.getElementById('replyCount').value;
    const favoriteCount = document.getElementById('favoriteCount').value;
    const hashtags = document.getElementById('hashtags').value;
    const urls = document.getElementById('urls').value;
    const mentions = document.getElementById('mentions').value;

    // Validate input data
    if (!post) {
        alert('Please fill in the post input field.');
        // Hide loading animation and revert button text
        this.classList.remove('loading');
        this.textContent = 'Check';
        return;
    }

    const data = {
        post: post,
        replyCount: replyCount,
        favoriteCount: favoriteCount,
        hashtags: hashtags,
        urls: urls,
        mentions: mentions
    };

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        const output = document.getElementById('output');
        output.innerHTML = `
            Sentiment: ${data.sentiment} (${data.sentiment_percent}%)<br>
            Sarcasm: ${data.sarcasm} (${data.sarcasm_percent}%)<br>
            Account Status: ${data.authenticity} (${data.authenticity_p}%)<br>
        `;
        output.style.display = 'block'; // Make the output visible
        // Hide loading animation and revert button text
        this.classList.remove('loading');
        this.textContent = 'Check';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
        // Hide loading animation and revert button text
        this.classList.remove('loading');
        this.textContent = 'Check';
    });
});
