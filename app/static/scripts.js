document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData(this);
    const uploadStatus = document.getElementById('upload-status');

    fetch('/resume/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        uploadStatus.style.display = 'block';
        if (data.error) {
            uploadStatus.textContent = data.error;
            uploadStatus.style.color = 'red';
        } else {
            uploadStatus.textContent = 'File uploaded successfully!';
            uploadStatus.style.color = 'green';
        }
    })
    .catch(error => {
        uploadStatus.style.display = 'block';
        uploadStatus.textContent = 'An error occurred!';
        uploadStatus.style.color = 'red';
    });
});

document.getElementById('train-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const trainStatus = document.getElementById('train-status');

    fetch('/resume/train', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        trainStatus.style.display = 'block';
        if (data.error) {
            trainStatus.textContent = data.error;
            trainStatus.style.color = 'red';
        } else {
            trainStatus.textContent = 'Model trained successfully! \n' + data.report; // Assuming server sends a summary report
            trainStatus.style.color = 'green';
        }
    })
    .catch(error => {
        trainStatus.style.display = 'block';
        trainStatus.textContent = 'An error occurred during training!';
        trainStatus.style.color = 'red';
    });
});