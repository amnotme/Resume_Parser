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