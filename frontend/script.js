document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('imageInput');


    fileInput.addEventListener('change', function() {
        const fileName = fileInput.files[0] ? fileInput.files[0].name : 'Choose Image';
        document.querySelector('.file-label').textContent = fileName;
    });


    document.getElementById('uploadForm').addEventListener('submit', async function(e) {
        e.preventDefault();

        const formData = new FormData(this);
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorMessage = `Error ${response.status}: ${response.statusText}`;
            console.error(errorMessage);
            return;
        }

        const resultHtml = await response.text();

        const resultWindow = window.open('', '_blank');
        resultWindow.document.open();
        resultWindow.document.write(resultHtml);
        resultWindow.document.close();
    });
});
