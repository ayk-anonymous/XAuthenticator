document.getElementById('check_button').addEventListener('click', function() {
    var twitterHandle = document.getElementById('twitter_handle').value;
    if (twitterHandle) {
        // This is where you would integrate with an API to check authenticity
        document.getElementById('output').textContent = "Checking " + twitterHandle + "...";
        
        // Simulate an API call response
        setTimeout(() => {
            document.getElementById('output').textContent = "Result for " + twitterHandle + ": " + (Math.random() > 0.5 ? "Real" : "Fake");
        }, 2000);
    } else {
        document.getElementById('output').textContent = "Please enter a Twitter handle.";
    }
});
