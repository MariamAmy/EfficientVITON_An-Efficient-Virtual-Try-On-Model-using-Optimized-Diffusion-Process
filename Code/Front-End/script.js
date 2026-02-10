const images = document.querySelectorAll('.image');
let currentImageIndex = 1;

function changeImage() {
  images.forEach((img, index) => {
    img.classList.toggle('active', index === currentImageIndex); // Add 'active' only to the current image
  });

  currentImageIndex = (currentImageIndex + 1) % images.length; // Cycle through images
}

setInterval(changeImage, 3000); // Change image every 3 seconds

const fileInput = document.getElementById("file-input");
const uploadArea = document.getElementById("upload-area");
const browseLink = document.getElementById("browse-link");


function encodeImage(file) {
  return new Promise((resolve, reject) => {
    try {
      const reader = new FileReader();

      // Read the file as a DataURL (base64-encoded string)
      reader.onload = (event) => {
        const base64String = event.target.result.split(",")[1]; // Extract only base64 content
        resolve(base64String);
      };

      reader.onerror = () => {
        reject("Error reading the file.");
      };

      // Read the file
      reader.readAsDataURL(file);
    } catch (error) {
      reject(`Error encoding image: ${error.message}`);
    }
  });
}


// Update file input change listener
fileInput.addEventListener("change", (e) => {
  const files = e.target.files;
  handleFileUpload(files);
});

// Drag-and-drop functionality
uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  const files = e.dataTransfer.files; // Get files from drag-and-drop
  handleFileUpload(files);
});

// Browse button trigger
browseLink.addEventListener("click", (e) => {
  e.preventDefault();
  fileInput.click(); // Open the file dialog
});

async function handleFileUpload(files) {
  if (files.length > 0) {
    const file = files[0];

    // Validate the file type
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file.");
      return;
    }

    // Validate the file size (10 MB limit)
    if (file.size > 10 * 1024 * 1024) {
      alert("File size exceeds the 10 MB limit.");
      return;
    }

    try {
      // Encode the image into base64
      const encodedImage = await encodeImage(file);

      if (file) {
        // Process the file, like converting to base64 or image object if needed
        const reader = new FileReader();
        reader.onload = (e) => {
          const image = e.target.result; // this could be a data URL, or other image object
          replaceUploadInterface(image); // Send directly to replaceUploadInference
        };
        reader.readAsDataURL(file); // Or use readAsArrayBuffer, depending on your needs
      }

      // Prepare the request body
      const requestBody = {
        image: encodedImage,
        filename: "test",
      };

      // Send the POST request in the background
      fetch("https://exact-marmoset-stable.ngrok-free.app/preprocess", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
          }
          console.log("API request successful:", response.status);
        })
        .catch((error) => {
          console.error("Error in API request:", error);
        })
        .finally(() => {
          // Hide the loader when the response arrives (whether success or error)
          hideLoader();
        });
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred while processing the image. Please try again.");
    }
  }
}

function replaceUploadInterface(imageSrc) {
  const images2 = document.querySelectorAll('.image2');
  images2.forEach((img, index) => {
    img.classList.toggle('active', index === 1); // Add 'active' only to the current image
  });

  uploadArea.innerHTML = `
    <div class="content-wrapper">
        <div class="uploaded-image-container">
            <img src="${imageSrc}" alt="Uploaded Image">
            <div class="loader-overlay" id="loader-overlay">
                <span class="loader" id="loading-spinner"></span>
            </div>
            <p class="image-caption">YOUR IMAGE</p>
        </div>
        <div class="vertical-line"></div>
        <div class="clothes-section">
            <h2>Choose Your Outfit</h2>
            <div class="clothes-container">
                <img src="imgs/03604_00.jpg" alt="Clothes 1" class="clothes-image" data-name="1">
                <img src="imgs/00599_00.jpg" alt="Clothes 9" class="clothes-image" data-name="9">
                <img src="imgs/00713_00.jpg" alt="Clothes 10" class="clothes-image" data-name="10">
                <img src="imgs/00888_00.jpg" alt="Clothes 11" class="clothes-image" data-name="11">
                <img src="imgs/05624_00.jpg" alt="Clothes 2" class="clothes-image" data-name="2">
                <img src="imgs/02365_00.jpg" alt="Clothes 12" class="clothes-image" data-name="12">
                <img src="imgs/02673_00.jpg" alt="Clothes 13" class="clothes-image" data-name="13">
                <img src="imgs/03922_00.jpg" alt="Clothes 14" class="clothes-image" data-name="14">
                <img src="imgs/07622_00.jpg" alt="Clothes 3" class="clothes-image" data-name="3">
                <img src="imgs/07627_00.jpg" alt="Clothes 15" class="clothes-image" data-name="15">
                <img src="imgs/08008_00.jpg" alt="Clothes 16" class="clothes-image" data-name="16">
                <img src="imgs/08212_00.jpg" alt="Clothes 17" class="clothes-image" data-name="17">
                <img src="imgs/10698_00.jpg" alt="Clothes 18" class="clothes-image" data-name="18">
                <img src="imgs/08314_00.jpg" alt="Clothes 4" class="clothes-image" data-name="4">
                <img src="imgs/11450_00.jpg" alt="Clothes 19" class="clothes-image" data-name="19">
                <img src="imgs/13102_00.jpg" alt="Clothes 20" class="clothes-image" data-name="20">
                <img src="imgs/14358_00.jpg" alt="Clothes 21" class="clothes-image" data-name="21">
                <img src="imgs/08432_00.jpg" alt="Clothes 5" class="clothes-image" data-name="5">
                <img src="imgs/09276_00.jpg" alt="Clothes 22" class="clothes-image" data-name="22">
                <img src="imgs/09290_00.jpg" alt="Clothes 23" class="clothes-image" data-name="23">
                <img src="imgs/08877_00.jpg" alt="Clothes 6" class="clothes-image" data-name="6">
                <img src="imgs/11715_00.jpg" alt="Clothes 24" class="clothes-image" data-name="24">
                <img src="imgs/11738_00.jpg" alt="Clothes 25" class="clothes-image" data-name="25">
                <img src="imgs/13468_00.jpg" alt="Clothes 26" class="clothes-image" data-name="26">
                <img src="imgs/09246_00.jpg" alt="Clothes 7" class="clothes-image" data-name="7">
                <img src="imgs/13844_00.jpg" alt="Clothes 27" class="clothes-image" data-name="27">
                <img src="imgs/10915_00.jpg" alt="Clothes 28" class="clothes-image" data-name="28">
                <img src="imgs/10567_00.jpg" alt="Clothes 29" class="clothes-image" data-name="29">
                <img src="imgs/09685_00.jpg" alt="Clothes 30" class="clothes-image" data-name="30">
                <img src="imgs/09256_00.jpg" alt="Clothes 8" class="clothes-image" data-name="8">
                <img src="imgs/09236_00.jpg" alt="Clothes 31" class="clothes-image" data-name="31">
                <img src="imgs/11584_00.jpg" alt="Clothes 32" class="clothes-image" data-name="32">
            </div>
            <div class="cloloader-overlay" id="cloloader-overlay">
                  <span class="cloloader" id="cloloading-spinner"></span>
            </div>
        </div>
    </div>
  `;

  const clothesImages = document.querySelectorAll('.clothes-image');
  clothesImages.forEach((clothingImage) => {
    clothingImage.addEventListener('click', () => {
      const clothesImageName = clothingImage.getAttribute('data-name'); // Get the name of the clicked clothes image
      handleClothesClick(imageSrc, clothesImageName); // Pass both imageSrc and clothesImageName
    });
  });
}

function showLoader() {
  const loaderOverlay = document.getElementById('loader-overlay');
  console.log(loaderOverlay);
  if (loaderOverlay) {
    loaderOverlay.style.display = 'flex'; // Show the loader overlay
  }
}

function hideLoader() {
  const loaderOverlay = document.getElementById('loader-overlay');
  if (loaderOverlay) {
    loaderOverlay.style.display = 'none'; // Hide the loader overlay
  }
}

function base64ToFile(base64, filename) {
  const arr = base64.split(','); // Split the base64 string
  const mime = arr[0].match(/:(.*?);/)[1]; // Extract the MIME type
  const bstr = atob(arr[1]); // Decode the base64 string
  let n = bstr.length;
  const u8arr = new Uint8Array(n); // Create a Uint8Array from the decoded string
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new File([u8arr], filename, { type: mime }); // Create and return the File object
}

function toBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = (error) => reject(error);
  });
}

async function handleClothesClick(srcImage,clothesImageName) {
  try {
    showLoading();
  
    // Send a POST request to the API
    const requestBody = {
      cloth_name: clothesImageName,
    };

    const response = await fetch('https://exact-marmoset-stable.ngrok-free.app/inference', {
      method: 'POST',
      headers: {
      'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    console.log(response);

    // Handle the response
    if (!response.ok) {
      throw new Error('Failed to fetch data from the server');
    }

    const data = await response.json();
    console.log("Response data:", data); // Log the response data

    // Assuming the API returns the encoded image in string
    const encodedImage = data.image;
    const returnedImageSrc = `data:image/png;base64,${encodedImage}`;

    // Update the UI with the returned image
    updateUploadAreaWithResult(srcImage ,returnedImageSrc);
  } catch (error) {
    console.error('Error:', error);
    alert('An error occurred while processing your request. Please try again.');
  } finally {
    // Hide the loading indicator
    hideLoading();
  }
}

function showLoading() {
  const loadingElement = document.getElementById('cloloader-overlay');
  console.log(loadingElement);
  if (loadingElement) {
    loadingElement.style.display = 'flex'; // Make the spinner visible
  }
}

function hideLoading() {
  const loadingElement = document.getElementById('cloloader-overlay');
  if (loadingElement) {
    loadingElement.style.display = 'none'; // Hide the spinner
  }
}


function updateUploadAreaWithResult(srcImage ,returnedImageSrc) {
  document.querySelector('.image-container-try').style.display = 'none';
  document.querySelector('.upload-area').style.height = '500px';
  const rounded_window = document.getElementById('rounded-window');
  const uploadArea = document.getElementById('upload-area');
  uploadArea.innerHTML = `
    <div class="uploadedd-image-container">
        <img src="${returnedImageSrc}" alt="Returned Image" class="centered-image">
    </div>
  `;
  // Create buttons container
  const buttonsContainer = document.createElement('div');
  buttonsContainer.style.display = 'flex';
  buttonsContainer.style.justifyContent = 'center';
  buttonsContainer.style.gap = '20px';
  buttonsContainer.style.marginTop = '20px';

    // Create Retry button
  const retryButton = document.createElement('button');
  retryButton.textContent = 'Retry';
  retryButton.style.padding = '5px 50px';
  retryButton.style.fontSize = '16px';
  retryButton.style.cursor = 'pointer';
  retryButton.style.backgroundColor = '#FFFFFF';
  retryButton.style.color = '#000000';
  retryButton.style.border = '1px solid #FFFFFF';
  retryButton.style.borderRadius = '8px';


  // Add click event for Retry button
  retryButton.addEventListener('click', () => {
    rounded_window.innerHTML = `
    <div class="header-row">
                        <h2>
                            <span style="color: #79BCCF;">TRY</span> VIRTUAL TRY ON
                        </h2>
                        <a href="main.html" class="exit-icon" aria-label="Exit">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" class="thin-x">
                                <line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
                                <line x1="6" y1="18" x2="18" y2="6" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
                            </svg>
                        </a>                        
                    </div>
      <div class="upload-area" id="upload-area">
      <div class="content-wrapper">
          <div class="uploaded-image-container">
              <img src="${srcImage}" alt="Uploaded Image">
              <div class="loader-overlay" id="loader-overlay">
                  <span class="loader" id="loading-spinner"></span>
              </div>
              <p class="image-caption">YOUR IMAGE</p>
          </div>
          <div class="vertical-line"></div>
          <div class="clothes-section">
            <h2>Choose Your Outfit</h2>
            <div class="clothes-container">
                <img src="imgs/03604_00.jpg" alt="Clothes 1" class="clothes-image" data-name="1">
                <img src="imgs/00599_00.jpg" alt="Clothes 9" class="clothes-image" data-name="9">
                <img src="imgs/00713_00.jpg" alt="Clothes 10" class="clothes-image" data-name="10">
                <img src="imgs/00888_00.jpg" alt="Clothes 11" class="clothes-image" data-name="11">
                <img src="imgs/05624_00.jpg" alt="Clothes 2" class="clothes-image" data-name="2">
                <img src="imgs/02365_00.jpg" alt="Clothes 12" class="clothes-image" data-name="12">
                <img src="imgs/02673_00.jpg" alt="Clothes 13" class="clothes-image" data-name="13">
                <img src="imgs/03922_00.jpg" alt="Clothes 14" class="clothes-image" data-name="14">
                <img src="imgs/07622_00.jpg" alt="Clothes 3" class="clothes-image" data-name="3">
                <img src="imgs/07627_00.jpg" alt="Clothes 15" class="clothes-image" data-name="15">
                <img src="imgs/08008_00.jpg" alt="Clothes 16" class="clothes-image" data-name="16">
                <img src="imgs/08212_00.jpg" alt="Clothes 17" class="clothes-image" data-name="17">
                <img src="imgs/10698_00.jpg" alt="Clothes 18" class="clothes-image" data-name="18">
                <img src="imgs/08314_00.jpg" alt="Clothes 4" class="clothes-image" data-name="4">
                <img src="imgs/11450_00.jpg" alt="Clothes 19" class="clothes-image" data-name="19">
                <img src="imgs/13102_00.jpg" alt="Clothes 20" class="clothes-image" data-name="20">
                <img src="imgs/14358_00.jpg" alt="Clothes 21" class="clothes-image" data-name="21">
                <img src="imgs/08432_00.jpg" alt="Clothes 5" class="clothes-image" data-name="5">
                <img src="imgs/09276_00.jpg" alt="Clothes 22" class="clothes-image" data-name="22">
                <img src="imgs/09290_00.jpg" alt="Clothes 23" class="clothes-image" data-name="23">
                <img src="imgs/08877_00.jpg" alt="Clothes 6" class="clothes-image" data-name="6">
                <img src="imgs/11715_00.jpg" alt="Clothes 24" class="clothes-image" data-name="24">
                <img src="imgs/11738_00.jpg" alt="Clothes 25" class="clothes-image" data-name="25">
                <img src="imgs/13468_00.jpg" alt="Clothes 26" class="clothes-image" data-name="26">
                <img src="imgs/09246_00.jpg" alt="Clothes 7" class="clothes-image" data-name="7">
                <img src="imgs/13844_00.jpg" alt="Clothes 27" class="clothes-image" data-name="27">
                <img src="imgs/10915_00.jpg" alt="Clothes 28" class="clothes-image" data-name="28">
                <img src="imgs/10567_00.jpg" alt="Clothes 29" class="clothes-image" data-name="29">
                <img src="imgs/09685_00.jpg" alt="Clothes 30" class="clothes-image" data-name="30">
                <img src="imgs/09256_00.jpg" alt="Clothes 8" class="clothes-image" data-name="8">
                <img src="imgs/09236_00.jpg" alt="Clothes 31" class="clothes-image" data-name="31">
                <img src="imgs/11584_00.jpg" alt="Clothes 32" class="clothes-image" data-name="32">
            </div>
            <div class="cloloader-overlay" id="cloloader-overlay">
                  <span class="cloloader" id="cloloading-spinner"></span>
            </div>
        </div>
      </div>
      </div>
      <div class="image-container-try">
                        <img id="image1" class="image2" src="imgs/Property1.png" alt="Rotating Image 1">
                        <img id="image2" class="image2 active" src="imgs/Property2.png" alt="Rotating Image 2">
                        <img id="image3" class="image2" src="imgs/Property3.png" alt="Rotating Image 3">
      </div>
    `;
    hideLoader();
    // Rebind click listeners for clothes selection
    const clothesImages = document.querySelectorAll('.clothes-image');
    clothesImages.forEach((clothingImage) => {
      clothingImage.addEventListener('click', () => {
        const clothesImageName = clothingImage.getAttribute('data-name');
        handleClothesClick(srcImage ,clothesImageName);
      });
    });
  });

  // Create Exit button
  const exitButton = document.createElement('button');
  exitButton.textContent = 'Exit';
  exitButton.style.padding = '5px 56px';
  exitButton.style.fontSize = '16px';
  exitButton.style.cursor = 'pointer';
  exitButton.style.backgroundColor = '#FFFFFF';
  exitButton.style.color = '#000000';
  exitButton.style.border = '1px solid #FFFFFF';
  exitButton.style.borderRadius = '8px';
  exitButton.addEventListener('click', () => {
    window.location.href = 'main.html'; // Redirect to main.html
  });

  // Append buttons to the buttons container
  buttonsContainer.appendChild(retryButton);
  buttonsContainer.appendChild(exitButton);

  // Append buttons container below the upload-area in the rounded-window
  const roundedWindow = document.querySelector('.rounded-window');
  roundedWindow.appendChild(buttonsContainer);
}